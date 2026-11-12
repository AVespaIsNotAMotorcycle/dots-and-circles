'''
Scores the page reader on real pages: each image in real_pages/ named
pageNNN (.png or .jpg) with a transcription pageNNN.txt beside it, one line
per column of Manchu, in Abkai. Columns of Chinese aren't transcribed.

The reader's text and the transcription are compared without punctuation,
and without footnote marks such as [1]. Punctuation is scored apart, by
punctuation_counts: how many of the transcription's commas and full stops
the reader found in the same line. Each transcribed line is matched to the found
line closest to it, so a line found in the wrong place or a column split
in two still counts, and found lines left over (other text read as Manchu,
usually) are listed. Scores are character and word accuracy:
1 - edit distance / length, over all lines.

Run with python -m ocr.evaluate_real_pages [--model "Run 31"]
[--draw FOLDER]. The model is a name from server/config.json, read with
its box classifier.
'''

import argparse
import os
import re

from PIL import Image

from ocr.evaluate_pages import load_reader
from ocr.box_classifier import load_classifier
from ocr.page_reader import draw_page, page_text, read_page
from ocr.train import edit_distance

directory = os.path.dirname(os.path.abspath(__file__))
PAGES_DIRECTORY = os.path.join(directory, "real_pages")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
# Punctuation in transcriptions, left out of the text's scores
PUNCTUATION = ",.᠈᠉()"
# The punctuation scored on its own, with the Manchu marks counted as
# their Latin ones
MARKS = {",": ",", ".": ".", "\u1808": ",", "\u1809": "."}


def clean(line):
    # A line in lower case (some transcriptions capitalize titles, which
    # Abkai doesn't), without punctuation or footnote marks ([1]), its
    # words separated by single spaces
    line = re.sub(r"\[\d+\]", " ", line.lower())
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

def pages(folder=PAGES_DIRECTORY):
    # (name, image path, transcription path) for each transcribed page
    found = []
    for name in sorted(os.listdir(folder)):
        stem, extension = os.path.splitext(name)
        transcription = os.path.join(folder, stem + ".txt")
        if extension.lower() in IMAGE_EXTENSIONS and os.path.exists(
                transcription):
            found.append((stem, os.path.join(folder, name), transcription))
    return found

def read_transcription(path):
    # A transcription's lines, as written, without blank ones
    with open(path, "r", encoding="utf-8") as file:
        return [line.rstrip("\n") for line in file if line.strip()]

def punctuation_counts(true, found):
    '''
    (marks in the true line, marks in the found line, marks matched) for a
    pair of lines with their punctuation: a mark is matched if the other
    line has one like it, wherever it is in the line.
    '''
    def counts(line):
        marks = [MARKS[character] for character in line
                 if character in MARKS]
        return {mark: marks.count(mark) for mark in set(MARKS.values())}
    true_counts, found_counts = counts(true), counts(found)
    return (sum(true_counts.values()), sum(found_counts.values()),
            sum(min(true_counts[mark], found_counts[mark])
                for mark in true_counts))

def score_page(page, transcription):
    '''
    How read_page's page compares with a transcription (its lines):
    {"characters": accuracy, "words": accuracy, "extra lines": [...],
    "pairs": [(true, found)], "punctuation": (true, found, matched)}, the
    pairs without punctuation and the punctuation counted over all lines
    (see punctuation_counts).
    '''
    true_lines = [line for line in transcription if clean(line)]
    found_lines = [line for line in page_text(page, abkai=True).split("\n")
                   if line.strip()]
    true = [clean(line) for line in true_lines]
    found = [clean(line) for line in found_lines]
    pairs, extra = match_lines(found, true)
    characters, errors, words, word_errors = score(pairs)
    # Each found line as read, by its cleaned text, to count its marks
    as_read = {}
    for clean_line, line in zip(found, found_lines):
        as_read.setdefault(clean_line, []).append(line)
    punctuation = [0, 0, 0]
    for true_line, (_, found_line) in zip(true_lines, pairs):
        read = as_read[found_line].pop(0) if as_read.get(found_line) else ""
        punctuation = [total + count for total, count in zip(
            punctuation, punctuation_counts(true_line, read))]
    return {"characters": 1 - errors / max(1, characters),
            "words": 1 - word_errors / max(1, words),
            "extra lines": extra, "pairs": pairs,
            "punctuation": tuple(punctuation)}

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model",
                        help="a model's name in server/config.json (by "
                             "default, its default_model)")
    parser.add_argument("--classifier",
                        help="box classifier weights (by default, the "
                             "config's)")
    parser.add_argument("--draw", help="a folder to save each page drawn "
                                       "with its boxes in")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print every line")
    args = parser.parse_args()
    (model, crop, whiten, stem_breaks), classifier_path = load_reader(
        args.model)
    classifier = load_classifier(args.classifier or classifier_path)

    totals = [0, 0, 0, 0]
    marks = [0, 0, 0]
    for name, image_path, transcription_path in pages():
        with Image.open(image_path) as image:
            page = read_page(image.convert("RGB"), model, classifier, crop,
                             whiten, stem_breaks)
        result = score_page(page, read_transcription(transcription_path))
        pairs, extra = result["pairs"], result["extra lines"]
        counts = score(pairs)
        totals = [total + count for total, count in zip(totals, counts)]
        marks = [total + count for total, count in
                 zip(marks, result["punctuation"])]
        true_marks, found_marks, matched = result["punctuation"]
        print(f"{name}: {result['characters']:.1%} characters, "
              f"{result['words']:.1%} words, {len(extra)} extra lines, "
              f"punctuation {matched}/{true_marks} found, "
              f"{found_marks - matched} extra")
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
              f"{1 - totals[3] / totals[2]:.1%} words, punctuation "
              f"{marks[2]}/{marks[0]} found, {marks[1] - marks[2]} extra")

if __name__ == "__main__":
    main()
