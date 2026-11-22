# Run from the repository root with:
#   .venv/bin/python -m ocr.diagnose ocr/runs/run033/weights_D-12-64_Adam_3e-3.npz --depth 12

'''
Diagnoses a ModelD on the real word crops without training anything, to
see where its errors come from:

- robustness: each crop read again moved, slanted or bent sideways inside
  the WIDTH-pixel strip (see VARIANTS), to see what the first layer is
  sensitive to;
- letter confusions: each reading aligned letter by letter with its label;
- crop shape: errors by the crop's height per letter, before scaling, to
  see whether wide crops (flourishes, slant) are read worse.

Writes report.md, robustness.png and shape.png to --out (by default a
diagnosis folder beside the weights).
'''

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import time
from collections import Counter
from multiprocessing import Pool

import numpy as np
from PIL import Image

from ocr import cursive, manchuai, photos
from ocr.modelD import ModelD
from ocr.photos import MANCHU_TO_ROMAN, manchu_to_roman, roman_to_manchu
from ocr.preprocess import crop_to_ink, stretch
from ocr.train import (CONTEXT_LENGTH, TOKEN_DIM, classes_in, image_to_array,
                       predict, read_words, token_count)
from ocr.training_data import WIDTH

# Each worker reads with its own model; about 300 MB each with 16 crops
BATCH_SIZE = 16
WORKERS = 8
# The Manchu comma and full stop, which ManchuAI's labels never have
PUNCTUATION = "᠈᠉"
# The robustness variants. "plain" is the usual reading. The others scale
# the crop to SMALL pixels wide, leaving room in the strip, so they're
# compared with "centred", which is only that. A half-pixel shift blurs
# like the slants and bends' sub-pixel shifts do, without moving anything.
SMALL = 24
PERIOD = 48     # rows per wave of a bend, about five or six letters
VARIANTS = {
    "plain": None,
    "centred": ("move", 0),
    "half pixel": ("move", 0.5),
    "moved 4px": ("move", 4),
    "slant ±2": ("slant", 2),
    "slant ±4": ("slant", 4),
    "slant ±8": ("slant", 8),
    "bend ±1": ("bend", 1),
    "bend ±2": ("bend", 2),
    "bend ±3": ("bend", 3),
    "bend ±4": ("bend", 4),
}

model = None


def sources():
    # name: [(path, label)] for each set of crops that's here
    found = {}
    if photos.available():
        found["photo"] = photos.set_images("benchmark")
    if cursive.available("v2"):
        found["cursive_v2"] = cursive.set_images("benchmark")
    if cursive.available("v1"):
        found["cursive_v1"] = cursive.all_images("v1")
    for name in manchuai.SETS:
        if manchuai.available(name):
            found[f"manchuai_{name}"] = manchuai.all_images(name)
    return found

def init_worker(weights, depth, emb_dim):
    global model
    classes = classes_in(weights)
    model = ModelD(token_dim=TOKEN_DIM, context_length=CONTEXT_LENGTH,
                   num_classes=classes, depth=depth, blank=classes - 1,
                   emb_dim=emb_dim)
    model.load_weights(weights)
    model.set_training(False)

def raw_crop(path):
    # As ocr.preprocess.prepare does it, before scaling to WIDTH
    return crop_to_ink(stretch(Image.open(path).convert("L"), whiten=True))

def variant(image, name):
    '''
    The crop (from raw_crop) as the model reads it in VARIANTS[name]: each
    row shifted sideways by an offset, linearly interpolated, on white.
    '''
    if VARIANTS[name] is None:
        height = max(1, round(image.height * WIDTH / image.width))
        return image.resize((WIDTH, height), Image.LANCZOS)
    kind, size = VARIANTS[name]
    height = max(1, round(image.height * SMALL / image.width))
    small = np.asarray(image.resize((SMALL, height), Image.LANCZOS),
                       dtype=np.float64)
    rows = np.arange(height)
    middle = (WIDTH - SMALL) / 2
    if kind == "move":
        offsets = np.full(height, middle + size)
    elif kind == "slant":
        offsets = middle + size * (rows / max(1, height - 1) - 0.5)
    else:
        offsets = middle + size * np.sin(2 * np.pi * rows / PERIOD)
    columns, source = np.arange(WIDTH), np.arange(SMALL)
    out = np.array([np.interp(columns - offset, source, row,
                              left=255, right=255)
                    for offset, row in zip(offsets, small)])
    return Image.fromarray(np.round(out).astype(np.uint8))

def read_batch(job):
    # Every variant's readings, romanized, for one batch of one source
    source, batch = job
    crops = [raw_crop(path) for path, _ in batch]
    rows = [{"source": source, "label": label, "width": crop.width,
             "height": crop.height}
            for (_, label), crop in zip(batch, crops)]
    for name in VARIANTS:
        images = [variant(crop, name) for crop in crops]
        entries = [(image_to_array(image), None, token_count(image))
                   for image in images]
        _, y = predict(model, entries)
        for row, guess in zip(rows, read_words(model, y, entries)):
            if source.startswith("manchuai_"):
                guess = "".join(c for c in guess if c not in PUNCTUATION)
            row[name] = manchu_to_roman(guess.strip())
    return rows

def letters(word):
    '''
    A word in the dataset's romanization as its Manchu letters, each
    spelled in that romanization (so ng is one letter).
    '''
    try:
        return [MANCHU_TO_ROMAN[letter] for letter in roman_to_manchu(word)
                if MANCHU_TO_ROMAN[letter]]
    except ValueError:
        return list(word)

def align(true, guess):
    '''
    Aligns two letter lists with the fewest edits. Returns the edit
    distance and the pairs (true letter or "", read letter or ""), so ("",
    x) is an inserted letter and (x, "") a dropped one.
    '''
    n, m = len(true), len(guess)
    cost = np.zeros((n + 1, m + 1), dtype=int)
    cost[:, 0], cost[0, :] = range(n + 1), range(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost[i, j] = min(cost[i - 1, j] + 1, cost[i, j - 1] + 1,
                             cost[i - 1, j - 1]
                             + (true[i - 1] != guess[j - 1]))
    pairs, i, j = [], n, m
    while i or j:
        if i and j and cost[i, j] == cost[i - 1, j - 1] + (
                true[i - 1] != guess[j - 1]):
            pairs.append((true[i - 1], guess[j - 1]))
            i, j = i - 1, j - 1
        elif i and cost[i, j] == cost[i - 1, j] + 1:
            pairs.append((true[i - 1], ""))
            i -= 1
        else:
            pairs.append(("", guess[j - 1]))
            j -= 1
    return cost[n, m], pairs[::-1]

def scores(rows, name):
    # (share of words exact, letter error rate) of the rows' name readings
    exact = np.mean([row[name] == row["label"] for row in rows])
    errors = sum(align(letters(row["label"]), letters(row[name]))[0]
                 for row in rows)
    return exact, errors / max(1, sum(len(letters(row["label"]))
                                      for row in rows))

def robustness_table(rows, names):
    lines = ["| set | crops | " + " | ".join(VARIANTS) + " |",
             "|---" * (len(VARIANTS) + 2) + "|"]
    for source in names:
        sub = [row for row in rows if row["source"] == source]
        cells = [f"{exact:.1%}" for exact, _ in
                 (scores(sub, name) for name in VARIANTS)]
        lines.append(f"| {source} | {len(sub)} | " + " | ".join(cells)
                     + " |")
    return lines

def confusion_lines(rows, title, count=15):
    shown = lambda letter: {"": "∅", " ": "␣"}.get(letter, letter)
    confusions, seen = Counter(), Counter()
    for row in rows:
        true = letters(row["label"])
        seen.update(true)
        for pair in align(true, letters(row["plain"]))[1]:
            if pair[0] != pair[1]:
                confusions[pair] += 1
    total = sum(confusions.values())
    lines = [f"### {title}", "",
             f"{total} letter errors over {sum(seen.values())} letters "
             "(∅ on the right: dropped; on the left: inserted).", "",
             "| true → read | count | share |", "|---|---|---|"]
    for (true, read), number in confusions.most_common(count):
        lines.append(f"| {shown(true)} → {shown(read)} | {number} | "
                     f"{number / max(1, total):.1%} |")
    lines += ["", "| letter | seen | misread |", "|---|---|---|"]
    rates = [(sum(n for (t, _), n in confusions.items() if t == letter)
              / number, letter, number)
             for letter, number in seen.items() if number >= 20]
    for rate, letter, number in sorted(rates, reverse=True)[:10]:
        lines.append(f"| {shown(letter)} | {number} | {rate:.1%} |")
    return lines + [""]

def shape_table(rows, names):
    '''
    Letter error rates by fifths of height ÷ width ÷ letters, the widest
    crops for their length first; and the series for the chart.
    '''
    lines = ["| set | widest | 2 | 3 | 4 | narrowest |", "|---" * 6 + "|"]
    series = {}
    for source in names:
        sub = [row for row in rows if row["source"] == source
               and letters(row["label"])]
        shape = np.array([row["height"] / row["width"]
                          / len(letters(row["label"])) for row in sub])
        order = np.argsort(shape, kind="stable")
        fifths = np.array_split(order, 5)
        series[source] = [scores([sub[i] for i in fifth], "plain")[1]
                          for fifth in fifths]
        lines.append(f"| {source} | " + " | ".join(
            f"{rate:.1%}" for rate in series[source]) + " |")
    return lines, series

def chart(series, labels, title, ylabel, path):
    # A line per source, in the default categorical colours, saved as PNG
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    colours = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
               "#008300"]
    surface, muted = "#fcfcfb", "#52514e"
    figure, axes = plt.subplots(figsize=(8, 4.2), facecolor=surface)
    axes.set_facecolor(surface)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    axes.grid(axis="y", color="#e4e3df")
    axes.set_axisbelow(True)
    xs = range(len(labels))
    for colour, (name, values) in zip(colours, series.items()):
        axes.plot(xs, values, color=colour, linewidth=2, marker="o",
                  markersize=6, markeredgecolor=surface, label=name)
    axes.set_xticks(list(xs), labels, rotation=30, ha="right")
    axes.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    axes.set_ylabel(ylabel, color=muted)
    axes.set_title(title, loc="left")
    axes.legend(frameon=False, fontsize=9, loc="upper left",
                bbox_to_anchor=(1.02, 1))
    figure.tight_layout()
    figure.savefig(path, dpi=150, facecolor=surface)
    plt.close(figure)

def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("weights", help="a ModelD weights .npz file")
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--out", help="folder for the report (default: "
                        "diagnosis/ beside the weights)")
    args = parser.parse_args()
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(
        args.weights)), "diagnosis")
    os.makedirs(out, exist_ok=True)

    found = sources()
    jobs = [(source, images[start:start + BATCH_SIZE])
            for source, images in found.items()
            for start in range(0, len(images), BATCH_SIZE)]
    rows = []
    start = time.time()
    with Pool(args.workers, initializer=init_worker,
              initargs=(args.weights, args.depth, args.emb_dim)) as pool:
        for number, batch_rows in enumerate(pool.imap_unordered(read_batch,
                                                                jobs), 1):
            rows += batch_rows
            if number % 50 == 0:
                print(f"{number} of {len(jobs)} batches read", flush=True)
    print(f"Read {len(rows)} crops in {time.time() - start:.0f}s.")

    names = list(found)
    report = [f"# Diagnosis of {os.path.relpath(args.weights)}", "",
              "## Robustness", "",
              "Words read exactly. All but plain scale the crop to "
              f"{SMALL}px wide in the {WIDTH}px strip, so compare them with "
              "centred; the half-pixel shift shows what the sub-pixel "
              "blurring of the slants and bends costs alone. A slant "
              "drifts the given pixels from top to bottom; a bend weaves "
              f"them either side, {PERIOD} rows per wave.", ""]
    report += robustness_table(rows, names) + ["", "## Letter confusions",
                                               ""]
    if "photo" in found:
        report += confusion_lines([row for row in rows
                                   if row["source"] == "photo"], "Print")
    for source in names:
        if source != "photo":
            report += confusion_lines([row for row in rows
                                       if row["source"] == source], source)
    lines, shape_series = shape_table(rows, names)
    report += ["## Crop shape", "",
               "Letter error rate by fifths of each set's crops, sorted by "
               "height ÷ width ÷ letters before scaling: the widest crops "
               "for their length first.", ""] + lines
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as file:
        file.write("\n".join(report) + "\n")

    robust_series = {source: [scores([row for row in rows
                                      if row["source"] == source], name)[0]
                              for name in VARIANTS] for source in names}
    chart(robust_series, list(VARIANTS), "Words read exactly, by variant",
          "Words read exactly", os.path.join(out, "robustness.png"))
    chart(shape_series, ["widest", "2", "3", "4", "narrowest"],
          "Letter errors by crop shape", "Letter error rate",
          os.path.join(out, "shape.png"))
    print(f"Wrote {out}/report.md, robustness.png and shape.png")

if __name__ == "__main__":
    main()
