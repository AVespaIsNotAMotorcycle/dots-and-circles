# Run from the repository root with:
#   .venv/bin/python -m server.select_synthetic
#
# Renders a fixed set of synthetic words into server/synthetic, for
# GET /random?source=synthetic to choose from. The fonts aren't in the
# repository or on the server, so the images are made here and committed.
#
# The words are random words from the test split (see ocr/split.py), so
# models trained on that split have never seen them, each in a random font.
# Each is augmented to look like a photographed print, as in training, and
# saved before preprocessing, so the server prepares it exactly as it does a
# photo.

import argparse
import json
import os
import shutil

import numpy as np

from ocr.photos import manchu_to_roman
from ocr.preprocess import prepare
from ocr.train import WORD_COUNT, generate_seeds
from ocr.training_data import fonts, lexicon, render
from ocr.training_data.augment import (ATTEMPTS, ROWS_PER_TOKEN, photograph,
                                       tokens_needed)

directory = os.path.dirname(os.path.abspath(__file__))
SYNTHETIC_DIRECTORY = os.path.join(directory, "synthetic")
# The augmentation epoch the images are made with. Test words are never
# trained on, so any epoch would do.
EPOCH = 0

def photographed(word, seed):
    '''
    The photo-like image augment() would make for this seed and EPOCH,
    before preprocessing, or None if it would fall back to the clean image
    (too short for CTC once prepared).
    '''
    canvas = render(word, fonts[seed // WORD_COUNT])
    minimum_height = tokens_needed(word) * ROWS_PER_TOKEN
    for attempt in range(ATTEMPTS):
        image = photograph(canvas, np.random.default_rng([seed, EPOCH,
                                                          attempt]))
        if prepare(image).height >= minimum_height:
            return image
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Renders some random test-split words into "
        "server/synthetic.")
    parser.add_argument("--words", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not fonts:
        raise SystemExit("No fonts in ocr/training_data/fonts")

    _, test_seeds = generate_seeds(training_photos=False)
    # Every test seed is a (word, font) pair; group them by word
    by_word = {}
    for seed in sorted(test_seeds):
        by_word.setdefault(seed % WORD_COUNT, []).append(seed)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(sorted(by_word))

    if os.path.isdir(SYNTHETIC_DIRECTORY):
        shutil.rmtree(SYNTHETIC_DIRECTORY)
    os.makedirs(SYNTHETIC_DIRECTORY)
    manifest = []
    for index in order:
        if len(manifest) == args.words:
            break
        word = lexicon[index]
        # Suffixes (᠊ᠨ and so on) are left out, as for the photos
        if word.startswith("᠊"):
            continue
        seed = int(rng.choice(by_word[index]))
        image = photographed(word, seed)
        if image is None:
            continue
        path = f"{len(manifest):03d}.png"
        image.save(os.path.join(SYNTHETIC_DIRECTORY, path))
        font = os.path.splitext(os.path.basename(
            fonts[seed // WORD_COUNT]))[0]
        manifest.append({"label": manchu_to_roman(word), "manchu": word,
                         "font": font, "seed": seed, "path": path})
    with open(os.path.join(SYNTHETIC_DIRECTORY, "manifest.json"), "w",
              encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=1)
    print(f"Rendered {len(manifest)} words to {SYNTHETIC_DIRECTORY}")

if __name__ == "__main__":
    main()
