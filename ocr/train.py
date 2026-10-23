# Run from the repository root with: python -m ocr.train
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from loss.CrossEntropyLoss import CrossEntropyLoss
from optim.SGD import SGD
from ocr.modelA import ModelA
from ocr.training_data import WIDTH, fonts, generate, lexicon

FONT_COUNT = len(fonts)
WORD_COUNT = len(lexicon)
CONTEXT_LENGTH = 256

directory = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(directory, "training_data", "alphabet.json"),
          "r", encoding="utf-8") as file:
    alphabet = json.load(file)
letter_indices = {letter: index for index, letter in enumerate(alphabet)}
# The extra character that marks the end of a word
END = len(alphabet)
NUM_CLASSES = len(alphabet) + 1

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

def image_to_array(image):
    '''
    Converts a word image to an array of shape (CONTEXT_LENGTH, WIDTH), where
    each row is one token. Ink is high and background is 0. Short images are
    padded with blank rows; tall ones are scaled down to fit.
    '''
    if image.height > CONTEXT_LENGTH:
        image = image.resize((WIDTH, CONTEXT_LENGTH), Image.LANCZOS)
    array = np.zeros((CONTEXT_LENGTH, WIDTH), dtype=np.uint8)
    array[:image.height] = 255 - np.asarray(image, dtype=np.uint8)
    return array

def generate_batches(seeds, batch_size=64):
    '''
    Converts seeds into words and images, and returns them as an array of
    batches.

    Batch: array of length batch_size, where each item is an Entry.
    Entry: tuple of form (Image, Label).
    Image: PIL Image converted to a grayscale array,
        made with training_data/__init__.py
    Label: An array of one-hot vectors, each one representing a character
        in the word.
    '''
    batches = []
    for start in range(0, len(seeds), batch_size):
        print(f"Generating batch: seeds {start} to {start+batch_size}.")
        batch = []
        for seed in seeds[start:start + batch_size]:
            image, word = generate(seed)
            batch.append((image_to_array(image), word_to_array(word)))
        batches.append(batch)
    return batches

def train_on_batch(model, loss_function, optimizer, batch, train=False):
    images = np.array([image for image, _ in batch]) / 255

    # Step 1: Forward Propogate
    y = model(images)

    # Step 2: Calculate Accuracy and Loss
    # Output token i predicts character i, so only the first len(label)
    # tokens (the word and its end marker) are scored. The rest are
    # labelled -1, which the loss ignores.
    labels = np.full(np.shape(y)[:-1], -1)
    for item, (_, label) in enumerate(batch):
        labels[item, :len(label)] = np.argmax(label, axis=-1)
    scored = labels >= 0

    predictions = np.argmax(y, axis=-1)
    accuracy = np.mean(predictions[scored] == labels[scored])
    if not train:
        loss = loss_function.loss(images, y, labels)
        return accuracy, loss

    # Step 3: Calculate Gradients
    loss, grads = loss_function(images, y, labels)

    # Step 4: Optimize
    model.descend(optimizer(grads))
    return accuracy, loss

def log_batch(epoch, batch_number, train_accuracy, train_loss,
              test_accuracy, test_loss):
    print(f"Epoch {epoch + 1}, batch {batch_number:>4} | "
          f"train accuracy {train_accuracy:.3f}, loss {train_loss:.4f} | "
          f"test accuracy {test_accuracy:.3f}, loss {test_loss:.4f}")

def moving_average(values, window):
    window = min(window, len(values))
    return np.convolve(values, np.ones(window) / window, mode="valid")

def graph(train_history, test_history, window=5):
    train_history = np.array(train_history)
    test_history = np.array(test_history)

    figure, (accuracy_axis, loss_axis) = plt.subplots(2, 1, sharex=True)
    for history, label in ((train_history, "train"), (test_history, "test")):
        # Each point averages a batch with the window - 1 batches before it
        steps = history[min(window, len(history)) - 1:, 0]
        accuracy_axis.plot(steps, moving_average(history[:, 1], window),
                           label=label)
        loss_axis.plot(steps, moving_average(history[:, 2], window),
                       label=label)
    accuracy_axis.set_ylabel("Accuracy")
    loss_axis.set_ylabel("Loss")
    loss_axis.set_xlabel("Batch")
    accuracy_axis.legend()
    plt.show()

def train(epochs = 3, learn_rate = 0.001):
    train_set, test_set = generate_seeds()

    print("Generating batches...")
    train_set = generate_batches(train_set)
    test_set = generate_batches(test_set)
    batch_count = len(train_set) + len(test_set)

    model = ModelA(context_length=CONTEXT_LENGTH, num_classes=NUM_CLASSES)
    loss_function = CrossEntropyLoss(model)
    optimizer = SGD(learn_rate)

    train_history = []
    test_history = []
    for epoch in range(epochs):
        train_index = 0
        test_index = 0
        for batch_number in range(batch_count):
            # Batches from each set are spaced across the whole epoch, so
            # whichever set runs out first leaves the rest to the other.
            step = epoch * batch_count + batch_number
            train_left = train_index < len(train_set)
            test_left = test_index < len(test_set)
            if test_left and (batch_number % 10 > 7 or not train_left):
                batch = test_set[test_index]
                accuracy, loss = train_on_batch(
                    model, loss_function, optimizer, batch, train=False)
                test_history.append([step, accuracy, loss])
                test_index += 1
            else:
                batch = train_set[train_index]
                accuracy, loss = train_on_batch(
                    model, loss_function, optimizer, batch, train=True)
                train_history.append([step, accuracy, loss])
                train_index += 1
            # Nothing to report until both sets have been run at least once
            if batch_number % 10 == 0 and train_history and test_history:
                test_numbers = test_history[-1]
                _, test_accuracy, test_loss = test_numbers

                train_numbers = train_history[-1]
                _, train_accuracy, train_loss = train_numbers

                log_batch(epoch, batch_number, train_accuracy, train_loss,
                    test_accuracy, test_loss)

    graph(train_history, test_history)


if __name__ == "__main__":
    train(epochs=1)
