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

from ocr.photos import DATA_DIRECTORY, spell_in_manchu

directory = os.path.dirname(os.path.abspath(__file__))
PHOTO_DIRECTORY = os.path.join(directory, "photos")

def main():
    parser = argparse.ArgumentParser(
        description="Copies one photo each of some random words into "
        "server/photos.")
    parser.add_argument("--words", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

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
        manifest.append({"label": label, "manchu": spell_in_manchu(label),
                         "path": path, "source": f"{label}/{name}"})
    with open(os.path.join(PHOTO_DIRECTORY, "manifest.json"), "w",
              encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=1)
    print(f"Copied {len(manifest)} photos to {PHOTO_DIRECTORY}")

if __name__ == "__main__":
    main()
