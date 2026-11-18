'''
Downloads the two handwritten test sets of mic7ch's ManchuAI-OCR (Chung &
Choi, arXiv 2507.06761) from Hugging Face into ocr/manchuai_data/
(gitignored), one folder per word as for the print photos (see
ocr.manchuai):

- 218: https://huggingface.co/datasets/mic7ch/test_data_long, the paper's
  218-word test set (its train and validation splits together).
- 753: https://huggingface.co/datasets/mic7ch/manchu, the test split (753
  words) of the repository's newer dataset. Only the test split is
  fetched: the dataset's real training part is the project's own photo set.

Neither dataset states a license, so the images are never committed.

The images are stored rotated 90° counter-clockwise, so they're turned
back upright here. Folders are named in the project's romanization (see
ocr.photos.MANCHU_TO_ROMAN), from each image's Manchu label, with one
change: the datasets write fi as ᠸᡳ (arawi for arafi), and Manchu has no
wi, so it's spelled ᡶᡳ.

Both are only published as parquet, which needs pyarrow (not a project
dependency: install it anywhere, e.g. in a throwaway venv with numpy and
Pillow, and run this with that Python). Run from the repository root:
    python -m ocr.training_data.fetch_manchuai
'''

import io
import os
import shutil
import urllib.request

from PIL import Image

from ocr.manchuai import DATA_DIRECTORY
from ocr.photos import manchu_to_roman

URL = "https://huggingface.co/datasets/mic7ch/{}/resolve/main/data/{}"
SOURCES = {
    "218": [("test_data_long", "train-00000-of-00001.parquet"),
            ("test_data_long", "validation-00000-of-00001.parquet")],
    "753": [("manchu", "test-00000-of-00001.parquet")],
}


def download(url):
    with urllib.request.urlopen(url) as response:
        return response.read()

def label_of(manchu):
    '''
    The project's romanization of a dataset label, None if it has letters
    with no romanization. ᠸᡳ (wi) is read as ᡶᡳ (fi).
    '''
    manchu = manchu.strip().replace("ᠸᡳ", "ᡶᡳ")
    try:
        return manchu_to_roman(manchu)
    except KeyError as error:
        print(f"Skipping {manchu!r}: no romanization for {error}")
        return None

def fetch(name):
    try:
        import pyarrow.parquet
    except ImportError:
        raise SystemExit("The ManchuAI sets are parquet, so this needs "
                         "pyarrow")
    count = 0
    for dataset, file_name in SOURCES[name]:
        table = pyarrow.parquet.read_table(io.BytesIO(download(
            URL.format(dataset, file_name))))
        for row in table.to_pylist():
            label = label_of(row["manchu"])
            if label is None:
                continue
            image = Image.open(io.BytesIO(row["im"]["bytes"]))
            # Stored rotated 90° counter-clockwise: turn it back clockwise
            image = image.rotate(-90, expand=True)
            folder = os.path.join(DATA_DIRECTORY, name, label)
            os.makedirs(folder, exist_ok=True)
            image.save(os.path.join(folder, f"{count:04d}.png"))
            count += 1
    print(f"{name}: {count} images")


def main():
    if os.path.isdir(DATA_DIRECTORY):
        shutil.rmtree(DATA_DIRECTORY)
    for name in SOURCES:
        fetch(name)


if __name__ == "__main__":
    main()
