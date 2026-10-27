# Run from the repository root with:
#   .venv/bin/python -m server.select_photos
#
# Copies a fixed subset of ocr/non_synthetic_data into server/photos, one
# photo per word, for GET /random to choose from. The full set is too big to
# deploy, and isn't in the repository.

import argparse
import json
import os
import random
import shutil

from ocr.photos import DATA_DIRECTORY, MANCHU_TO_ABKAI, manchu_to_abkai
from ocr.training_data import lexicon

directory = os.path.dirname(os.path.abspath(__file__))
PHOTO_DIRECTORY = os.path.join(directory, "photos")

# Manchu for each Abkai spelling, for words not in the lexicon. Where several
# letters share a spelling (ᡳ and ᠢ, say), the Manchu one comes first in
# MANCHU_TO_ABKAI, so it's the one kept.
ABKAI_TO_MANCHU = {}
for letter, abkai in MANCHU_TO_ABKAI.items():
    if abkai:
        ABKAI_TO_MANCHU.setdefault(abkai, letter)
LONGEST_SPELLING = max(len(abkai) for abkai in ABKAI_TO_MANCHU)

def abkai_to_manchu(word):
    '''
    Transliterates a word in Abkai to Manchu script, e.g. abka -> ᠠᠪᡴᠠ,
    reading the longest spelling it can at each step (so ng is ᠩ, not ᠨᡤ).
    '''
    manchu = ""
    position = 0
    while position < len(word):
        for length in range(LONGEST_SPELLING, 0, -1):
            letter = ABKAI_TO_MANCHU.get(word[position:position + length])
            if letter:
                manchu += letter
                position += length
                break
        else:
            raise ValueError(f"No Manchu for {word[position:]!r} in {word!r}")
    return manchu

def main():
    parser = argparse.ArgumentParser(
        description="Copies one photo each of some random words into "
        "server/photos.")
    parser.add_argument("--words", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # The lexicon's spelling where it has the word, since it knows which
    # letters each word really uses. Suffixes (᠊ᠨ and so on) are left out,
    # since a photo is of a whole word.
    spellings = {}
    for word in lexicon:
        if not word.startswith("᠊"):
            spellings.setdefault(manchu_to_abkai(word), word)

    rng = random.Random(args.seed)
    labels = rng.sample(sorted(os.listdir(DATA_DIRECTORY)), args.words)
    if os.path.isdir(PHOTO_DIRECTORY):
        shutil.rmtree(PHOTO_DIRECTORY)
    manifest = []
    os.makedirs(PHOTO_DIRECTORY)
    for label in sorted(labels):
        folder = os.path.join(DATA_DIRECTORY, label)
        name = rng.choice(sorted(os.listdir(folder)))
        # Saved under the (ASCII) label, since some of the dataset's file
        # names use other romanizations, e.g. dabarakū.png. The original
        # name is kept as source.
        path = label + os.path.splitext(name)[1].lower()
        shutil.copy(os.path.join(folder, name),
                    os.path.join(PHOTO_DIRECTORY, path))
        manchu = spellings.get(label) or abkai_to_manchu(label)
        manifest.append({"label": label, "manchu": manchu, "path": path,
                         "source": f"{label}/{name}"})
    with open(os.path.join(PHOTO_DIRECTORY, "manifest.json"), "w",
              encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=1)
    print(f"Copied {len(manifest)} photos to {PHOTO_DIRECTORY}")

if __name__ == "__main__":
    main()
