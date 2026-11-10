'''
Scores the page reader on real pages: each image in real_pages/ named
pageNNN (.png or .jpg) with a transcription pageNNN.txt beside it, one line
per column of Manchu, in Abkai. Columns of Chinese aren't transcribed.

The reader's text and the transcription are compared without punctuation
(the reader leaves it out). Each transcribed line is matched to the found
line closest to it, so a line found in the wrong place or a column split
in two still counts, and found lines left over (other text read as Manchu,
usually) are listed. Scores are character and word accuracy:
1 - edit distance / length, over all lines.

Run with python -m ocr.evaluate_real_pages [--model "Run 30"]
[--draw FOLDER]. The model is a name from server/config.json, read with
its box classifier.
'''

import argparse
import os

from PIL import Image

from ocr.evaluate_pages import load_reader
from ocr.box_classifier import load_classifier
from ocr.page_reader import draw_page, page_text, read_page
from ocr.train import edit_distance

directory = os.path.dirname(os.path.abspath(__file__))
PAGES_DIRECTORY = os.path.join(directory, "real_pages")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
# Punctuation in transcriptions, which the reader doesn't give
PUNCTUATION = ",.᠈᠉"


def clean(line):
    # A line without punctuation, its words separated by single spaces
    for mark in PUNCTUATION:
        line = line.replace(mark, " ")
    return " ".join(line.split())

def match_lines(found, true):
    '''
    Matches each true line to the found line with the fewest character
    errors, each found line used once, in the true lines' order. Returns
    [(true line, found line or "")] and the found lines left over.
    '''
    unused = list(range(len(found)))
    pairs = []
    for line in true:
        if not unused:
            pairs.append((line, ""))
            continue
        best = min(unused, key=lambda index: edit_distance(found[index],
                                                           line))
        unused.remove(best)
        pairs.append((line, found[best]))
    return pairs, [found[index] for index in unused]

def score(pairs):
    # (characters, character errors, words, word errors) over line pairs
    characters = errors = words = word_errors = 0
    for true, found in pairs:
        characters += len(true)
        errors += edit_distance(found, true)
        words += len(true.split())
        word_errors += edit_distance(found.split(), true.split())
    return characters, errors, words, word_errors

def pages():
    # (name, image path, transcription path) for each transcribed page
    found = []
    for name in sorted(os.listdir(PAGES_DIRECTORY)):
        stem, extension = os.path.splitext(name)
        transcription = os.path.join(PAGES_DIRECTORY, stem + ".txt")
        if extension.lower() in IMAGE_EXTENSIONS and os.path.exists(
                transcription):
            found.append((stem, os.path.join(PAGES_DIRECTORY, name),
                          transcription))
    return found

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="Run 30",
                        help="a model's name in server/config.json")
    parser.add_argument("--classifier",
                        help="box classifier weights (by default, the "
                             "config's)")
    parser.add_argument("--draw", help="a folder to save each page drawn "
                                       "with its boxes in")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print every line")
    args = parser.parse_args()
    (model, crop, whiten), classifier_path = load_reader(args.model)
    classifier = load_classifier(args.classifier or classifier_path)

    totals = [0, 0, 0, 0]
    for name, image_path, transcription_path in pages():
        with open(transcription_path, "r", encoding="utf-8") as file:
            true = [clean(line) for line in file if clean(line)]
        with Image.open(image_path) as image:
            page = read_page(image.convert("RGB"), model, classifier, crop,
                             whiten)
        found = [line for line in page_text(page, abkai=True).split("\n")
                 if line.strip()]
        pairs, extra = match_lines(found, true)
        counts = score(pairs)
        totals = [total + count for total, count in zip(totals, counts)]
        print(f"{name}: {1 - counts[1] / counts[0]:.1%} characters, "
              f"{1 - counts[3] / counts[2]:.1%} words, "
              f"{len(extra)} extra lines")
        if args.verbose:
            for true_line, found_line in pairs:
                print(f"  found: {found_line}\n  true:  {true_line}")
            for line in extra:
                print(f"  extra: {line}")
        if args.draw:
            os.makedirs(args.draw, exist_ok=True)
            draw_page(page).save(os.path.join(args.draw, name + ".png"))
    if totals[0]:
        print(f"All pages: {1 - totals[1] / totals[0]:.1%} characters, "
              f"{1 - totals[3] / totals[2]:.1%} words")

if __name__ == "__main__":
    main()
