'''
Rebuilds photo_words.json, the words in the photographed test data
(ocr/non_synthetic_data/). The photos are gitignored but the list isn't, so
the train/test split (ocr/split.py) is the same on every machine, with or
without the photos.

Run from the repository root, after the photo data changes:
    python -m ocr.training_data.list_photo_words
'''

import json
import os

from ocr.photos import DATA_DIRECTORY, available

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "photo_words.json")


def main():
    if not available():
        raise SystemExit(f"No photo data at {DATA_DIRECTORY}")
    words = sorted(name for name in os.listdir(DATA_DIRECTORY)
                   if os.path.isdir(os.path.join(DATA_DIRECTORY, name)))
    data = {
        "description": "The words (folder names) of the photographed "
                       "print words in ocr/non_synthetic_data/, in the "
                       "dataset's romanization (see "
                       "ocr.photos.MANCHU_TO_ROMAN)",
        "source": "https://www.scidb.cn/en/detail?dataSetId="
                  "b45491b63d694534a9323acf14846586",
        "license": "CC BY 4.0",
        "words": words,
    }
    with open(PATH, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=1)
        file.write("\n")
    print(f"Wrote {len(words)} words to {PATH}")


if __name__ == "__main__":
    main()
