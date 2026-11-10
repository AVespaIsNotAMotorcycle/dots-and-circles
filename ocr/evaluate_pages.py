'''
Scores the page reader on synthetic pages (training_data/pages.py), whose
true boxes are known:

- boxes: a found box matches a true one when they overlap with an
  intersection over union of at least MATCH_IOU, each used once. Precision
  is the share of found boxes that match, recall the share of true ones.
  Unmatched true boxes are counted as merged (inside a found box that
  covers another true box too), split (covered by several found boxes) or
  missed.
- kinds (with a classifier): the share of matched boxes given the right
  kind, Manchu or other.
- reading (with --model, which needs the classifier too): over matched
  Manchu boxes, the share read exactly and the mean character accuracy, in
  the dataset's romanization as in the run reports. And for whole pages,
  the character accuracy of the page's text (a line per column with
  Manchu, see training_data.pages.page_lines) against the true text, which
  counts every error of layout, classing and reading; and the seconds per
  page.

--model reads server/config.json for the model's class, weights and
preprocessing, and by default uses its box classifier.

Pages are seeded from --start, so the same arguments score the same pages.
'''

import argparse
import json
import os
import time

from ocr.box_classifier import label_words, load_classifier
from ocr.layout import find_words
from ocr.photos import manchu_to_roman
from ocr.training_data.pages import generate_page, page_lines

MATCH_IOU = 0.5


def iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    overlap = (right - left) * (bottom - top)
    area = lambda box: (box[2] - box[0]) * (box[3] - box[1])
    return overlap / (area(a) + area(b) - overlap)

def match(found, true):
    '''
    Pairs found and true boxes greedily by intersection over union, best
    first, keeping pairs of at least MATCH_IOU. Returns {found index: true
    index}.
    '''
    pairs = sorted(((iou(f, t), i, j) for i, f in enumerate(found)
                    for j, t in enumerate(true)), reverse=True)
    pairing, used = {}, set()
    for score, i, j in pairs:
        if score < MATCH_IOU:
            break
        if i not in pairing and j not in used:
            pairing[i] = j
            used.add(j)
    return pairing

def centre_inside(box, outer):
    x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return outer[0] <= x < outer[2] and outer[1] <= y < outer[3]

def unmatched_kinds(found, true, pairing):
    # Counts of unmatched true boxes that were merged, split or missed
    matched = set(pairing.values())
    counts = {"merged": 0, "split": 0, "missed": 0}
    for j, box in enumerate(true):
        if j in matched:
            continue
        holders = [f for f in found if centre_inside(box, f)]
        if any(sum(centre_inside(other, f) for other in true) > 1
               for f in holders):
            counts["merged"] += 1
        elif sum(iou(f, box) > 0 for f in found) > 1:
            counts["split"] += 1
        else:
            counts["missed"] += 1
    return counts

def score_page(seed, photo, layout=None, classifier=None, reader=None):
    '''
    Scores one page. reader is (model, crop, whiten) to read it too.
    '''
    if reader is not None:
        # Imported here: ocr.train needs matplotlib, which scoring layouts
        # doesn't
        from ocr.page_reader import page_text, read_page
        from ocr.train import character_accuracy, edit_distance
    image, items = generate_page(seed, layout=layout, photo=photo)
    start = time.perf_counter()
    if reader is not None:
        page = read_page(image, *reader[:1], classifier, *reader[1:])
    else:
        page = find_words(image)
        if classifier is not None:
            label_words(page, classifier)
    seconds = time.perf_counter() - start
    found = [page.to_input(word["box"]) for word in page.words]
    true = [item["box"] for item in items]
    pairing = match(found, true)
    result = {"found": len(found), "true": len(true),
              "matched": len(pairing),
              "angle": page.angle}
    result.update(unmatched_kinds(found, true, pairing))
    if classifier is not None:
        for kind in ("manchu", "other"):
            pairs = [(i, j) for i, j in pairing.items()
                     if items[j]["kind"] == kind]
            result[f"{kind} matched"] = len(pairs)
            result[f"{kind} right"] = sum(page.words[i]["kind"] == kind
                                          for i, _ in pairs)
    if reader is not None:
        read = [(manchu_to_roman(page.words[i]["text"]),
                 manchu_to_roman(items[j]["text"]))
                for i, j in pairing.items()
                if items[j]["kind"] == "manchu"
                and page.words[i]["kind"] == "manchu"]
        result["read"] = len(read)
        result["exact"] = sum(guess == word for guess, word in read)
        result["character"] = sum(character_accuracy(guess, word)
                                  for guess, word in read)
        true_text = page_lines(items)
        found_text = page_text(page)
        result["text characters"] = len(true_text)
        result["text errors"] = edit_distance(found_text, true_text)
        result["seconds"] = seconds
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pages", type=int, default=50)
    parser.add_argument("--start", type=int, default=100000,
                        help="the first page's seed")
    parser.add_argument("--photo", action="store_true",
                        help="score photographed-looking pages")
    parser.add_argument("--layout", help="score only this layout")
    parser.add_argument("--classifier",
                        help="a box classifier's weights, to score kinds "
                             "and split runs of Chinese characters")
    parser.add_argument("--model",
                        help="the name of a model in server/config.json, "
                             "to read the pages with")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print each page's scores")
    args = parser.parse_args()
    reader = None
    if args.model:
        reader, classifier_path = load_reader(args.model)
        args.classifier = args.classifier or classifier_path
    classifier = (load_classifier(args.classifier) if args.classifier
                  else None)

    totals = {}
    for seed in range(args.start, args.start + args.pages):
        result = score_page(seed, args.photo, args.layout, classifier,
                            reader)
        if args.verbose:
            print(seed, result)
        for key, value in result.items():
            totals[key] = totals.get(key, 0) + value
    print(f"{args.pages} pages, {totals['true']} true boxes, "
          f"{totals['found']} found")
    print(f"Precision {totals['matched'] / max(1, totals['found']):.1%}, "
          f"recall {totals['matched'] / max(1, totals['true']):.1%}")
    print(f"Unmatched true boxes: {totals['merged']} merged, "
          f"{totals['split']} split, {totals['missed']} missed")
    if classifier is not None:
        print("Kinds right: " + ", ".join(
            f"{kind} {totals[f'{kind} right'] / max(1, totals[f'{kind} matched']):.1%}"
            f" of {totals[f'{kind} matched']}" for kind in ("manchu", "other")))
    if reader is not None:
        read = max(1, totals["read"])
        print(f"Manchu words read: {totals['exact'] / read:.1%} exactly, "
              f"{totals['character'] / read:.1%} character accuracy "
              f"(of {totals['read']})")
        print(f"Page text: {1 - totals['text errors'] / max(1, totals['text characters']):.1%} "
              f"character accuracy; {totals['seconds'] / args.pages:.2f} s "
              f"a page")

def load_reader(name):
    '''
    A model from server/config.json, as (model, crop, whiten), and the
    path of the config's box classifier.
    '''
    # Imported here: the server needs Flask, which scoring layouts doesn't
    from server.app import CONFIG_PATH, load_model
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)
    root = os.path.dirname(CONFIG_PATH)
    entries = [entry for entry in config["models"] if entry["name"] == name]
    if not entries:
        raise SystemExit(f"No model named {name!r} in {CONFIG_PATH}.")
    entry = entries[0]
    preprocess = entry.get("preprocess", {})
    return ((load_model(entry, root), preprocess.get("crop", True),
             preprocess.get("whiten", True)),
            os.path.join(root, config["box_classifier"]))

if __name__ == "__main__":
    main()
