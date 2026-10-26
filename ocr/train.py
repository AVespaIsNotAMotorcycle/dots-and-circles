# Run from the repository root with: .venv/bin/python -m ocr.train

import os
# Each model trains on its own thread, so numpy's BLAS shouldn't start 16
# threads of its own per matrix multiply too; they'd fight over the cores.
# This only works if it's set before numpy is first imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import csv
import gc
import json
import random
import threading
import traceback

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from loss.CrossEntropyLoss import CrossEntropyLoss
from loss.CTCLoss import CTCLoss
from optim.SGD import SGD
from optim.Adam import Adam
from ocr.modelA import ModelA
from ocr.modelB import ModelB
from ocr.modelC import ModelC
from ocr.modelD import ModelD
from ocr.training_data import HEIGHT, WIDTH, fonts, generate, lexicon

FONT_COUNT = len(fonts)
WORD_COUNT = len(lexicon)
IMAGE_HEIGHT = HEIGHT
# Each token is a group of this many pixel rows, flattened
ROWS_PER_TOKEN = 4
TOKEN_DIM = ROWS_PER_TOKEN * WIDTH
CONTEXT_LENGTH = IMAGE_HEIGHT // ROWS_PER_TOKEN

directory = os.path.dirname(os.path.abspath(__file__))
# Each run gets its own numbered folder here, holding its log, weights and
# graph
RUNS_DIRECTORY = os.path.join(directory, "runs")
# epoch counts from 0 and includes progress through the current one, e.g.
# 2.5 is halfway through the third epoch
LOG_COLUMNS = ["model", "step", "epoch", "set", "accuracy", "loss"]
with open(os.path.join(directory, "training_data", "alphabet.json"),
          "r", encoding="utf-8") as file:
    alphabet = json.load(file)
letter_indices = {letter: index for index, letter in enumerate(alphabet)}
# The extra character that marks the end of a word. CTC models use this
# class as their blank instead.
END = len(alphabet)
NUM_CLASSES = len(alphabet) + 1
# Positions the longest label needs: its letters plus the end marker
MAX_LABEL_LENGTH = max(len(word) for word in lexicon) + 1

def generate_seeds(validation_set = False):
    train_proportion = 0.7
    valid_proportion = 0
    test_proportion = 0.3

    if validation_set:
        valid_proportion = 0.15
        test_proportion = 0.15

    train_set = []
    valid_set = []
    test_set = []

    seed_count = FONT_COUNT * WORD_COUNT
    for i in range(seed_count):
        seed = i
        lot = random.random()
        if lot < train_proportion:
            train_set.append(seed)
        elif lot < train_proportion + valid_proportion:
            valid_set.append(seed)
        else:
            test_set.append(seed)

    random.shuffle(train_set)
    random.shuffle(valid_set)
    random.shuffle(test_set)

    if validation_set:
        return train_set, valid_set, test_set

    return train_set, test_set

def word_to_array(word):
    '''
    Converts a word to an array of one-hot vectors. Each vector represents
    a character in that word. Each vector is of length n+1, where n is the
    number of characters in the alphabet (see training_data/alphabet.json)
    and the +1 corresponds to an extra character which indicates the end of
    the word. The array should always end with that character's one-hot
    vector, and that vector should never appear except as the final element
    of the array.
    '''
    array = np.zeros((len(word) + 1, NUM_CLASSES))
    for position, letter in enumerate(word):
        array[position][letter_indices[letter]] = 1
    array[-1][END] = 1
    return array

def array_to_word(array):
    '''
    Undoes word_to_array. Also accepts model outputs, in which case each
    position is read as its most likely character.
    '''
    word = ""
    for index in np.argmax(array, axis=-1):
        if index == END:
            break
        word += alphabet[index]
    return word

def ctc_decode(array, length, blank):
    '''
    Reads a CTC model's output for one image: the most likely class at each
    of its length real tokens, with repeats merged and blanks dropped.
    '''
    word = ""
    previous = None
    for index in np.argmax(array[:length], axis=-1):
        if index != previous and index != blank:
            word += alphabet[index]
        previous = index
    return word

def read_words(model, y, entries):
    # Decodes the model's outputs y for entries into words
    if getattr(model, "uses_ctc", False):
        return [ctc_decode(prediction, length, model.blank)
                for prediction, (_, _, length) in zip(y, entries)]
    return [array_to_word(prediction) for prediction in y]

def edit_distance(a, b):
    '''
    The fewest single-letter insertions, deletions and substitutions that
    turn a into b.
    '''
    previous = list(range(len(b) + 1))
    for i, letter_a in enumerate(a, start=1):
        current = [i]
        for j, letter_b in enumerate(b, start=1):
            current.append(min(previous[j] + 1,
                               current[j - 1] + 1,
                               previous[j - 1] + (letter_a != letter_b)))
        previous = current
    return previous[-1]

def character_accuracy(guess, word):
    # 1 for an exact reading, falling with each wrong letter, floored at 0
    return max(0, 1 - edit_distance(guess, word) / len(word))

def image_to_array(image):
    '''
    Converts a word image to an array of shape (CONTEXT_LENGTH, TOKEN_DIM),
    where each token is ROWS_PER_TOKEN pixel rows laid end to end. Ink is
    high and background is 0. Short images are padded with blank rows to
    IMAGE_HEIGHT; tall ones are scaled down to fit.
    '''
    if image.height > IMAGE_HEIGHT:
        image = image.resize((WIDTH, IMAGE_HEIGHT), Image.LANCZOS)
    array = np.zeros((IMAGE_HEIGHT, WIDTH), dtype=np.uint8)
    array[:image.height] = 255 - np.asarray(image, dtype=np.uint8)
    return array.reshape(CONTEXT_LENGTH, TOKEN_DIM)

def token_count(image):
    '''
    Number of tokens in image_to_array's output that hold part of the image
    rather than padding. A token that's only partly image counts as real.
    '''
    height = min(image.height, IMAGE_HEIGHT)
    return -(-height // ROWS_PER_TOKEN)

def generate_batches(seeds, batch_size=64):
    '''
    Converts seeds into words and images, and returns them as an array of
    batches.

    Batch: array of length batch_size, where each item is an Entry.
    Entry: tuple of form (Image, Label, Length).
    Image: PIL Image converted to a grayscale array,
        made with training_data/__init__.py
    Label: An array of one-hot vectors, each one representing a character
        in the word.
    Length: The number of tokens in Image that aren't padding.
    '''
    batches = []
    for start in range(0, len(seeds), batch_size):
        print(f"Generating batch: seeds {start} to {start+batch_size}.")
        batch = []
        for seed in seeds[start:start + batch_size]:
            image, word = generate(seed)
            batch.append((image_to_array(image), word_to_array(word),
                          token_count(image)))
        batches.append(batch)
    return batches

def predict(model, entries):
    # Runs the model on entries, masking each one's padding tokens
    images = np.array([image for image, _, _ in entries]) / 255
    model.set_lengths([length for _, _, length in entries])
    return images, model(images)

def train_on_batch(model, loss_function, optimizer, batch, train=False):
    # Dropout (if the model has any) only runs on training batches
    model.set_training(train)

    # Step 1: Forward Propogate
    images, y = predict(model, batch)

    # Step 2: Calculate Accuracy and Loss
    words = [array_to_word(label) for _, label, _ in batch]
    guesses = read_words(model, y, batch)
    accuracy = np.mean([character_accuracy(guess, word)
                        for guess, word in zip(guesses, words)])

    if getattr(model, "uses_ctc", False):
        # CTC labels are just the letters, with no end marker; the loss
        # only uses each image's real tokens
        labels = [list(np.argmax(label[:-1], axis=-1))
                  for _, label, _ in batch]
        extra = ([length for _, _, length in batch],)
    else:
        # Output token i predicts character i, so only the first
        # len(label) tokens (the word and its end marker) are scored. The
        # rest are labelled -1, which the loss ignores.
        labels = np.full(np.shape(y)[:-1], -1)
        for item, (_, label, _) in enumerate(batch):
            assert len(label) <= np.shape(y)[1], \
                f"A label needs {len(label)} positions, but the model " \
                f"only outputs {np.shape(y)[1]}"
            labels[item, :len(label)] = np.argmax(label, axis=-1)
        extra = ()

    if not train:
        loss = loss_function.loss(images, y, labels, *extra)
        return accuracy, loss

    # Step 3: Calculate Gradients
    loss, grads = loss_function(images, y, labels, *extra)

    # Step 4: Optimize
    model.descend(optimizer(grads))
    return accuracy, loss

def log_batch(name, epoch, batch_number, train_accuracy, train_loss,
              test_accuracy, test_loss):
    print(f"Model {name}, epoch {epoch + 1}, batch {batch_number:>4} | "
          f"train accuracy {train_accuracy:.3f}, loss {train_loss:.4f} | "
          f"test accuracy {test_accuracy:.3f}, loss {test_loss:.4f}")

def moving_average(values, window):
    window = min(window, len(values))
    return np.convolve(values, np.ones(window) / window, mode="valid")

def new_run_directory():
    '''
    Makes and returns ocr/runs/runNNN, numbered one past the highest run so
    far (counting the older graphs in run_graphs too).
    '''
    numbers = [0]
    for folder in (RUNS_DIRECTORY, os.path.join(directory, "run_graphs")):
        if os.path.isdir(folder):
            for name in os.listdir(folder):
                if name.startswith("run") and name[3:6].isdigit():
                    numbers.append(int(name[3:6]))
    run_directory = os.path.join(RUNS_DIRECTORY, f"run{max(numbers) + 1:03}")
    os.makedirs(run_directory)
    return run_directory

def weights_path(run_directory, name):
    # Run names can contain "/" and spaces, e.g. "C-8/128 Adam 1e-3"
    file_name = name.replace("/", "-").replace(" ", "_")
    return os.path.join(run_directory, f"weights_{file_name}.npz")

def read_log(run_directory):
    '''
    Reads a run's log.csv into {model: {"train": rows, "test": rows}}, where
    rows is an array of [epoch, accuracy, loss]. Logs from before the epoch
    column was added give the step in its place, so the second return value
    says which it is: "Epoch" or "Batch".
    '''
    histories = {}
    with open(os.path.join(run_directory, "log.csv"), newline="") as file:
        reader = csv.DictReader(file)
        x_column = "epoch" if "epoch" in reader.fieldnames else "step"
        for row in reader:
            sets = histories.setdefault(row["model"],
                                        {"train": [], "test": []})
            sets[row["set"]].append([float(row[x_column]),
                                     float(row["accuracy"]),
                                     float(row["loss"])])
    histories = {name: {key: np.array(rows) for key, rows in sets.items()}
                 for name, sets in histories.items()}
    return histories, "Epoch" if x_column == "epoch" else "Batch"

def graph(run_directory, window=5):
    '''
    Graphs a run from its log.csv, and saves the graph to the run's folder.
    Each model gets its own colour; train is solid and test is dashed.
    Works on any run folder, including one from a run that crashed.
    '''
    histories, x_label = read_log(run_directory)
    figure, (accuracy_axis, loss_axis) = plt.subplots(2, 1, sharex=True)
    for index, (name, sets) in enumerate(histories.items()):
        for label, style in (("train", "-"), ("test", "--")):
            history = sets[label]
            if len(history) == 0:
                continue
            # Each point averages a batch with the window - 1 batches before
            xs = history[min(window, len(history)) - 1:, 0]
            accuracy_axis.plot(xs, moving_average(history[:, 1], window),
                               style, color=f"C{index}",
                               label=f"{name} {label}")
            loss_axis.plot(xs, moving_average(history[:, 2], window),
                           style, color=f"C{index}", label=f"{name} {label}")
    accuracy_axis.set_ylabel("Accuracy")
    loss_axis.set_ylabel("Loss")
    loss_axis.set_xlabel(x_label)
    accuracy_axis.legend()
    figure.savefig(os.path.join(run_directory, "graph.png"))
    plt.show()

def show_predictions(models, batches, count=5):
    '''
    Prints count random words from batches alongside each model's reading of
    their images. models: {name: model}; all read the same words.
    '''
    entries = random.sample([entry for batch in batches for entry in batch],
                            count)
    guesses = {}
    for name, model in models.items():
        model.set_training(False)
        _, predictions = predict(model, entries)
        guesses[name] = read_words(model, predictions, entries)

    print("Word -> " + " | ".join(f"Model {name}" for name in models))
    for item, (_, label, _) in enumerate(entries):
        word = array_to_word(label)
        readings = []
        for name in models:
            guess = guesses[name][item]
            mark = "✓" if guess == word else "✗"
            readings.append(f"{mark} {guess or '(empty)'}")
        print(f"{word} -> " + " | ".join(readings))

def get_learn_rate(batch_count, initial_rate):
    # return initial_rate / ((0.001 * batch_count) + 1)
    return initial_rate

def train(epochs = 3):
    run_directory = new_run_directory()
    print(f"Saving this run to {run_directory}")
    run_models(run_directory, epochs)
    # run_models has returned, so its batches and models are unreachable;
    # collect them now, before graphing
    gc.collect()
    graph(run_directory)

def run_models(run_directory, epochs):
    '''
    Trains the models, one thread each, logging every batch's results to
    log.csv as it goes and saving each model's weights at the end of every
    epoch, so a crash loses at most the current epoch's training and none
    of the log. Everything it makes is local, so it's all freed when it
    returns.
    '''
    train_set, test_set = generate_seeds()

    print("Generating batches...")
    train_set = generate_batches(train_set)
    test_set = generate_batches(test_set)

    # Every model below is trained with every optimizer below, each pairing
    # as its own run, e.g. "D-8/64 Adam 1e-3". All runs train side by side,
    # each on its own thread, on the same batches in the same order. Both
    # dicts hold factories (lambdas), since each run needs its own fresh
    # model and optimizer.
    models = {
        # "A": lambda: ModelA(token_dim=TOKEN_DIM, context_length=CONTEXT_LENGTH,
        #             num_classes=NUM_CLASSES, depth=4),
        # "B": lambda: ModelB(token_dim=TOKEN_DIM, context_length=CONTEXT_LENGTH,
        #             num_classes=NUM_CLASSES, depth=4,
        #             num_readouts=MAX_LABEL_LENGTH),
        # "B-4": lambda: ModelB(token_dim=TOKEN_DIM, context_length=CONTEXT_LENGTH,
        #             num_classes=NUM_CLASSES, depth=4,
        #             num_readouts=MAX_LABEL_LENGTH),
        # "B-8": lambda: ModelB(token_dim=TOKEN_DIM, context_length=CONTEXT_LENGTH,
        #             num_classes=NUM_CLASSES, depth=8,
        #             num_readouts=MAX_LABEL_LENGTH),
        # "C": lambda: ModelC(token_dim=TOKEN_DIM, context_length=CONTEXT_LENGTH,
        #             num_classes=NUM_CLASSES, depth=4, blank=END),
        # "C-4": lambda: ModelC(token_dim=TOKEN_DIM, context_length=CONTEXT_LENGTH,
        #             num_classes=NUM_CLASSES, depth=4, blank=END),
        # "C-8": lambda: ModelC(token_dim=TOKEN_DIM, context_length=CONTEXT_LENGTH,
        #             num_classes=NUM_CLASSES, depth=8, blank=END),
        # "C-16": lambda: ModelC(token_dim=TOKEN_DIM, context_length=CONTEXT_LENGTH,
        #             num_classes=NUM_CLASSES, depth=16, blank=END),
        # "C-8/32": lambda: ModelC(token_dim=TOKEN_DIM, context_length=CONTEXT_LENGTH,
        #             num_classes=NUM_CLASSES, depth=8, blank=END, emb_dim=32),
        # "C-16/32": lambda: ModelC(token_dim=TOKEN_DIM, context_length=CONTEXT_LENGTH,
        #             num_classes=NUM_CLASSES, depth=16, blank=END, emb_dim=32),
        # "C-8/64": lambda: ModelC(token_dim=TOKEN_DIM, context_length=CONTEXT_LENGTH,
        #             num_classes=NUM_CLASSES, depth=8, blank=END, emb_dim=64),
        # "C-8/128": lambda: ModelC(token_dim=TOKEN_DIM, context_length=CONTEXT_LENGTH,
        #             num_classes=NUM_CLASSES, depth=8, blank=END, emb_dim=128),
        "D-8/64": lambda: ModelD(token_dim=TOKEN_DIM, context_length=CONTEXT_LENGTH,
                    num_classes=NUM_CLASSES, depth=8, blank=END, emb_dim=64,
                    drop_rate=0.1),
    }
    optimizers = {
        # "SGD 0.1": lambda: SGD(0.1, max_norm=5),
        # "Adam 3e-4": lambda: Adam(0.0003),
        # "Adam 1e-3": lambda: Adam(0.001),
        "Adam 3e-3": lambda: Adam(0.003),
    }

    runs = {}
    for model_name, make_model in models.items():
        for optimizer_name, make_optimizer in optimizers.items():
            model = make_model()
            optimizer = make_optimizer()
            if getattr(model, "uses_ctc", False):
                loss_function = CTCLoss(model, blank=model.blank)
            else:
                loss_function = CrossEntropyLoss(model)
            runs[f"{model_name} {optimizer_name}"] = {
                "model": model,
                "loss_function": loss_function,
                "optimizer": optimizer,
                # get_learn_rate's schedule starts from each optimizer's
                # own rate
                "learn_rate": optimizer.learn_rate,
                # Set by the run's thread if it crashes
                "failed": False,
            }

    # One log for every model. Threads write rows to it in whatever order
    # they finish batches, so the lock stops two rows being written at once.
    log_file = open(os.path.join(run_directory, "log.csv"), "w", newline="")
    log = csv.writer(log_file)
    log.writerow(LOG_COLUMNS)
    log_lock = threading.Lock()

    # Each model trains on its own thread, over the same batches in the same
    # order. numpy releases Python's GIL inside most array operations, so
    # the threads really do run at the same time for much of each batch.
    # They're daemon threads, so Ctrl+C ends the whole program rather than
    # waiting for them; the log is already on disk by then.
    threads = []
    for name, run in runs.items():
        thread = threading.Thread(
            target=train_model, daemon=True,
            args=(name, run, train_set, test_set, epochs,
                  run_directory, log, log_file, log_lock))
        thread.start()
        threads.append(thread)
    for thread in threads:
        # A join with a timeout, in a loop, so Ctrl+C still reaches this
        # (main) thread while it waits
        while thread.is_alive():
            thread.join(timeout=0.5)
    log_file.close()

    # Test words, so these are images the models haven't trained on.
    # Models whose thread crashed are left out.
    finished = {name: run["model"] for name, run in runs.items()
                if not run["failed"]}
    if finished:
        show_predictions(finished, test_set)

def epoch_schedule(train_set, test_set):
    '''
    Yields (batch_number, testing, batch) for one epoch. Batches from each
    set are spaced across the whole epoch, so whichever set runs out first
    leaves the rest to the other. Every model's thread gets the same order.
    '''
    train_index = 0
    test_index = 0
    for batch_number in range(len(train_set) + len(test_set)):
        train_left = train_index < len(train_set)
        test_left = test_index < len(test_set)
        testing = test_left and (batch_number % 10 > 7 or not train_left)
        if testing:
            batch = test_set[test_index]
            test_index += 1
        else:
            batch = train_set[train_index]
            train_index += 1
        yield batch_number, testing, batch

def train_model(name, run, train_set, test_set, epochs,
                run_directory, log, log_file, log_lock):
    '''
    Runs on its own thread: trains one model for every epoch, logging each
    batch and saving the weights after each epoch. An error stops only this
    model; it's printed, and the other models carry on.
    '''
    batch_count = len(train_set) + len(test_set)
    # The latest [accuracy, loss] from each set, for printing
    last = {"train": None, "test": None}
    try:
        for epoch in range(epochs):
            for batch_number, testing, batch in epoch_schedule(train_set,
                                                               test_set):
                step = epoch * batch_count + batch_number
                set_name = "test" if testing else "train"
                run["optimizer"].set_rate(get_learn_rate(step,
                                                         run["learn_rate"]))
                accuracy, loss = train_on_batch(
                    run["model"], run["loss_function"], run["optimizer"],
                    batch, train=not testing)

                with log_lock:
                    log.writerow([name, step,
                                  epoch + batch_number / batch_count,
                                  set_name, accuracy, loss])
                    # Write each batch to disk now, so a crash can't lose it
                    log_file.flush()

                last[set_name] = [accuracy, loss]
                # Nothing to report until both sets have been run once
                if batch_number % 10 == 0 and last["train"] and last["test"]:
                    log_batch(name, epoch, batch_number, *last["train"],
                              *last["test"])

            # Overwrites the last epoch's weights
            run["model"].save_weights(weights_path(run_directory, name))
    except Exception:
        run["failed"] = True
        print(f"Model {name} crashed; the other models will carry on.\n"
              f"{traceback.format_exc()}")

if __name__ == "__main__":
    train(epochs=30)
