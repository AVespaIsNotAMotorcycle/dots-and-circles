'''
Downloads mic7ch's handwritten cursive Manchu word crops from Hugging Face
into ocr/cursive_data/ (gitignored), one folder per word as for the print
photos (see ocr.cursive):

- v2: https://huggingface.co/datasets/mic7ch/cursive_v2, 1,940 words from
  the manuscript Talking Dead (preface and chapter 01). Split by word into
  train, monitor and benchmark (cursive_words.json).
- v1: https://huggingface.co/datasets/mic7ch/cursive_v1, 327 words from a
  1680 military document, in another hand. Only ever tested on.

Both are CC BY 4.0. Folders are named in the project's romanization (see
ocr.photos.MANCHU_TO_ROMAN), from each crop's Manchu label; the datasets'
own Möllendorff (with ū and š) isn't used, though it agrees. Both store
their crops rotated 90° counter-clockwise (v1's README doesn't say so), so
they're turned back upright here.

v2 is only published as parquet, which needs pyarrow (not a project
dependency: install it anywhere, e.g. in a throwaway venv, and run this
with that Python). Run from the repository root:
    python -m ocr.training_data.fetch_cursive
'''

import csv
import io
import os
import shutil
import urllib.parse
import urllib.request

from PIL import Image

from ocr.cursive import DATA_DIRECTORY
from ocr.photos import manchu_to_roman

V1_URL = "https://huggingface.co/datasets/mic7ch/cursive_v1/resolve/main/test/"
V2_URL = ("https://huggingface.co/datasets/mic7ch/cursive_v2/resolve/main/"
          "data/test-00000-of-00001.parquet")


def download(url):
    with urllib.request.urlopen(url) as response:
        return response.read()

def save(version, manchu, name, image):
    '''
    Saves one crop as <version>/<label>/<name>, returning the label, or
    None if the Manchu label has letters with no romanization.
    '''
    try:
        label = manchu_to_roman(manchu.strip())
    except KeyError as error:
        print(f"Skipping {name} ({manchu!r}): no romanization for {error}")
        return None
    folder = os.path.join(DATA_DIRECTORY, version, label)
    os.makedirs(folder, exist_ok=True)
    image.save(os.path.join(folder, os.path.basename(name)))
    return label

def fetch_v1():
    rows = csv.DictReader(io.StringIO(
        download(V1_URL + "metadata.csv").decode("utf-8")))
    count = 0
    for row in rows:
        image = Image.open(io.BytesIO(download(
            V1_URL + urllib.parse.quote(row["file_name"]))))
        # Stored rotated 90° counter-clockwise: turn it back clockwise
        image = image.rotate(-90, expand=True)
        if save("v1", row["manchu"], row["file_name"], image):
            count += 1
    print(f"v1: {count} crops")

def fetch_v2():
    try:
        import pyarrow.parquet
    except ImportError:
        raise SystemExit("cursive_v2 is parquet, so this needs pyarrow")
    table = pyarrow.parquet.read_table(io.BytesIO(download(V2_URL)))
    count = 0
    for row in table.to_pylist():
        image = Image.open(io.BytesIO(row["image"]["bytes"]))
        # Stored rotated 90° counter-clockwise: turn it back clockwise
        image = image.rotate(-90, expand=True)
        if save("v2", row["manchu"], row["image"]["path"], image):
            count += 1
    print(f"v2: {count} crops")


def main():
    if os.path.isdir(DATA_DIRECTORY):
        shutil.rmtree(DATA_DIRECTORY)
    fetch_v1()
    fetch_v2()


if __name__ == "__main__":
    main()
