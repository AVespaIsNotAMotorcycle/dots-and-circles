'''
Rebuilds photo_words.json, the words in the photographed data
(ocr/non_synthetic_data/), split into training, monitor and benchmark words
(see ocr.split.split_photo_words). The photos are gitignored but the list
isn't, so the train/test split (ocr/split.py) is the same on every machine,
with or without the photos.

Run from the repository root, after the photo data changes:
    python -m ocr.training_data.list_photo_words
'''

import json
import os

from ocr.photos import DATA_DIRECTORY, available
from ocr.split import split_photo_words

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "photo_words.json")


def main():
    if not available():
        raise SystemExit(f"No photo data at {DATA_DIRECTORY}")
    counts = {}
    for name in os.listdir(DATA_DIRECTORY):
        folder = os.path.join(DATA_DIRECTORY, name)
        if os.path.isdir(folder):
            counts[name] = len(os.listdir(folder))
    sets = split_photo_words(counts)
    data = {
        "description": "The words (folder names) of the photographed "
                       "print words in ocr/non_synthetic_data/, in the "
                       "dataset's romanization (see "
                       "ocr.photos.MANCHU_TO_ROMAN), split by "
                       "ocr.split.split_photo_words: train is trained on, "
                       "monitor is scored after every epoch, and benchmark "
                       "is the held-out test",
        "source": "https://www.scidb.cn/en/detail?dataSetId="
                  "b45491b63d694534a9323acf14846586",
        "license": "CC BY 4.0",
        "photos": {name: sum(counts[word] for word in words)
                   for name, words in sets.items()},
        **sets,
    }
    with open(PATH, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=1)
        file.write("\n")
    for name, words in sets.items():
        print(f"{name}: {len(words)} words, {data['photos'][name]} photos")
    print(f"Wrote {len(counts)} words to {PATH}")


if __name__ == "__main__":
    main()
