'''
Trains ocr.box_classifier's MLP on synthetic pages (training_data/pages.py):
each page is split by ocr.layout as a real one would be, and each found box
is labelled with the kind of the true items whose centres it holds. Boxes
holding both kinds, or none, are left out.

The pages use every single word of the lexicon, not only unseen ones: the
classifier only tells Manchu from other text, so it can see any word.
Short words are repeated SHORT_REPEATS times, since they're the hard case
(as square as Chinese characters) and some of the commonest words in print.

Run with python -m ocr.train_box_classifier [output.npz]. It prints the
validation accuracy for each kind, and for short Manchu words.
'''

import argparse
import multiprocessing
import os

import numpy as np

from loss.CrossEntropyLoss import CrossEntropyLoss
from ocr.box_classifier import KINDS, box_features, make_classifier
from ocr.evaluate_pages import centre_inside
from ocr.layout import find_words
from ocr.split import group_key
from ocr.training_data import lexicon
from ocr.training_data.pages import NIRUGU, generate_page
from optim.Adam import Adam
from utils.softmax import softmax

SHORT_LENGTH = 3      # letters, not counting invisible marks
SHORT_REPEATS = 20
# Seeds well away from evaluate_pages.py's (100000 on)
TRAIN_SEEDS = range(1000000, 1001000)
VALID_SEEDS = range(2000000, 2000150)
WORKERS = 6
EPOCHS = 15
BATCH_SIZE = 64
LEARN_RATE = 1e-3
DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "server", "weights", "box_classifier.npz")

_words = None


def page_words():
    global _words
    if _words is None:
        singles = sorted({word for entry in lexicon for word in entry.split()
                          if NIRUGU not in word})
        short = [word for word in singles
                 if len(group_key(word)) <= SHORT_LENGTH]
        _words = singles + short * SHORT_REPEATS
    return _words

def page_boxes(seed):
    '''
    The features, labels (indices into KINDS) and shortness (Manchu words
    of at most SHORT_LENGTH letters) of the boxes found on one page.
    '''
    image, items = generate_page(seed, photo=seed % 2 == 0,
                                 words=page_words())
    page = find_words(image)
    features, labels, short = [], [], []
    for word in page.words:
        box = page.to_input(word["box"])
        inside = [item for item in items if centre_inside(item["box"], box)]
        kinds = {item["kind"] for item in inside}
        if len(kinds) != 1:
            continue
        kind = kinds.pop()
        features.append(box_features(page.ink, word["box"], page.unit))
        labels.append(KINDS.index(kind))
        short.append(kind == "manchu" and len(inside) == 1
                     and len(group_key(inside[0]["text"])) <= SHORT_LENGTH)
    return features, labels, short

def make_set(seeds):
    with multiprocessing.Pool(WORKERS) as pool:
        results = pool.map(page_boxes, seeds, chunksize=10)
    features = np.array([f for result in results for f in result[0]])
    labels = np.array([l for result in results for l in result[1]])
    short = np.array([s for result in results for s in result[2]])
    return features, labels, short

def report(model, features, labels, short):
    model.set_training(False)
    guesses = np.argmax(softmax(model(features)), axis=-1)
    correct = guesses == labels
    parts = [f"{kind} {correct[labels == index].mean():.2%} "
             f"({(labels == index).sum()})"
             for index, kind in enumerate(KINDS)]
    parts.append(f"short Manchu {correct[short].mean():.2%} ({short.sum()})")
    return ", ".join(parts)

def main():
    parser = argparse.ArgumentParser(description="Train the box classifier.")
    parser.add_argument("output", nargs="?", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    np.random.seed(0)
    train = make_set(TRAIN_SEEDS)
    valid = make_set(VALID_SEEDS)
    print(f"Boxes: {len(train[1])} train, {len(valid[1])} validation")

    model = make_classifier()
    loss_function = CrossEntropyLoss(model)
    optimizer = Adam(LEARN_RATE)
    rng = np.random.default_rng(0)
    for epoch in range(EPOCHS):
        model.set_training(True)
        order = rng.permutation(len(train[1]))
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            x, labels = train[0][batch], train[1][batch]
            loss, grads = loss_function(x, model(x), labels)
            model.descend(optimizer(grads))
            losses.append(loss)
        print(f"Epoch {epoch + 1}: loss {np.mean(losses):.4f}; validation "
              f"{report(model, *valid)}")
    model.save_weights(args.output)
    print(f"Saved {args.output}")

if __name__ == "__main__":
    main()
