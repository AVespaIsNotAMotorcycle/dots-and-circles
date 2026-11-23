'''
Tunes the beam search's language model weight and letter bonus (see
ocr/beam_search.py) on the monitor sets, then scores the benchmarks with
and without it.

  .venv/bin/python -m ocr.tune_lm WEIGHTS --depth 12 [--workers 8] [--out DIR]

The model reads every image once (whitened and cropped, as the server
does); its outputs are kept in DIR/outputs_*.npz, so later runs only
decode. DIR is lm/ beside the weights by default; the report goes in
DIR/report.md.
'''
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import itertools
import time
from multiprocessing import Pool

import numpy as np

import ocr.evaluate_non_synthetic as evaluate
from ocr import cursive, manchuai
from ocr.beam_search import beam_search, log_softmax
from ocr.char_lm import load_lm, training_words
from ocr.photos import load_photo, manchu_to_roman, set_images
from ocr.train import (alphabet, ctc_decode, edit_distance, image_to_array,
                       predict, token_count)

WEIGHTS = [0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0]
BONUSES = [0, 0.5, 1, 1.5, 2, 3]
MONITORS = ["photo_monitor", "cursive_monitor"]
BENCHMARKS = ["photo", "cursive_v2", "cursive_v1", "manchuai_218",
              "manchuai_753"]


def images_of(name):
    return {"photo_monitor": lambda: set_images("monitor"),
            "cursive_monitor": lambda: cursive.set_images("monitor"),
            "photo": lambda: set_images("benchmark"),
            "cursive_v2": lambda: cursive.set_images("benchmark"),
            "cursive_v1": lambda: cursive.all_images("v1"),
            "manchuai_218": lambda: manchuai.all_images("218"),
            "manchuai_753": lambda: manchuai.all_images("753")}[name]()


def model_outputs(batch):
    # The model's outputs for a batch of (path, label), as the evaluator
    # prepares them
    images = [load_photo(path, True, True) for path, _ in batch]
    entries = [(image_to_array(image), None, token_count(image))
               for image in images]
    _, y = predict(evaluate.model, entries)
    return [(np.asarray(output, dtype=np.float32), length)
            for output, (_, _, length) in zip(y, entries)]


def outputs_for(name, pool, out):
    '''
    (labels, outputs, lengths) for a set, from the cache if it's there.
    '''
    path = os.path.join(out, f"outputs_{name}.npz")
    if os.path.exists(path):
        saved = np.load(path)
        return list(saved["labels"]), saved["outputs"], saved["lengths"]
    images = images_of(name)
    batches = [images[i:i + evaluate.BATCH_SIZE]
               for i in range(0, len(images), evaluate.BATCH_SIZE)]
    read = [pair for batch in pool.imap(model_outputs, batches)
            for pair in batch]
    labels = [label for _, label in images]
    outputs = np.stack([output for output, _ in read])
    lengths = np.array([length for _, length in read])
    np.savez(path, labels=np.array(labels), outputs=outputs, lengths=lengths)
    return labels, outputs, lengths


# Each decoding worker builds the language model once
lm = None


def init_decoder():
    global lm
    lm = load_lm()


def decode(job):
    # Readings of a chunk of outputs with one setting; weight None is the
    # greedy decoder
    outputs, lengths, blank, weight, bonus = job
    if weight is None:
        return [ctc_decode(output, length, blank)
                for output, length in zip(outputs, lengths)]
    return [beam_search(log_softmax(output[:length]), blank, alphabet, lm,
                        weight, bonus)
            for output, length in zip(outputs, lengths)]


def read_all(pool, outputs, lengths, blank, weight, bonus, chunk=40):
    jobs = [(outputs[i:i + chunk], lengths[i:i + chunk], blank, weight,
             bonus) for i in range(0, len(outputs), chunk)]
    return [reading for part in pool.map(decode, jobs) for reading in part]


def romanized(readings, name):
    # As the evaluator scores them: ManchuAI's labels have no punctuation
    if name.startswith("manchuai"):
        readings = ["".join(c for c in r if c not in evaluate.PUNCTUATION)
                    .strip() for r in readings]
    return [manchu_to_roman(r) for r in readings]


def scores(guesses, labels):
    exact = np.mean([g == l for g, l in zip(guesses, labels)])
    errors = sum(edit_distance(g, l) for g, l in zip(guesses, labels))
    return exact, 1 - errors / sum(len(l) for l in labels)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("weights")
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out")
    args = parser.parse_args()
    out = args.out or os.path.join(os.path.dirname(
        os.path.abspath(args.weights)), "lm")
    os.makedirs(out, exist_ok=True)

    data = {}
    start = time.time()
    with Pool(args.workers, initializer=evaluate.init_worker,
              initargs=(args.weights, args.depth, args.emb_dim)) as pool:
        for name in MONITORS + BENCHMARKS:
            data[name] = outputs_for(name, pool, out)
    # The blank is the last class (see train.classes_in)
    blank = data[MONITORS[0]][1].shape[-1] - 1
    print(f"Model outputs ready in {time.time() - start:.0f}s")

    report = [f"# Language model decoding: {args.weights}\n"]
    p = report.append
    with Pool(args.workers, initializer=init_decoder) as pool:
        # --- Tuning on the monitor sets
        grid = {}
        start = time.time()
        for weight, bonus in itertools.product(WEIGHTS, BONUSES):
            if weight == 0 and bonus:
                continue
            row = []
            for name in MONITORS:
                labels, outputs, lengths = data[name]
                guesses = romanized(read_all(pool, outputs, lengths, blank,
                                             weight, bonus), name)
                row.append(scores(guesses, labels))
            grid[(weight, bonus)] = row
            print(f"weight {weight}, bonus {bonus}: " + ", ".join(
                f"{n} {e:.1%}" for n, (e, _) in zip(MONITORS, row)),
                flush=True)
        greedy_monitor = []
        for name in MONITORS:
            labels, outputs, lengths = data[name]
            greedy_monitor.append(scores(romanized(read_all(
                pool, outputs, lengths, blank, None, 0), name), labels))
        print(f"Grid done in {time.time() - start:.0f}s")
        # Best by the two monitor sets' exact words, averaged
        best = max(grid, key=lambda k: (np.mean([e for e, _ in grid[k]]),
                                        -k[0]))

        p("## Tuning on the monitor sets\n")
        p("Words exact on " + " / ".join(MONITORS) + ". Greedy decoding: "
          + " / ".join(f"{e:.1%}" for e, _ in greedy_monitor)
          + f". Best: weight {best[0]}, bonus {best[1]}.\n")
        p("| weight \\ bonus | " + " | ".join(map(str, BONUSES)) + " |")
        p("|---" * (len(BONUSES) + 1) + "|")
        for weight in WEIGHTS:
            cells = [" / ".join(f"{e:.1%}" for e, _ in grid[(weight, b)])
                     if (weight, b) in grid else "" for b in BONUSES]
            p(f"| {weight} | " + " | ".join(cells) + " |")
        p("")

        # --- Benchmarks
        # Words the language model has seen, as an entry or inside one
        seen = {manchu_to_roman(word) for word in training_words()}
        seen |= {part for entry in seen for part in entry.split()}
        p("## Benchmarks\n")
        p(f"Words exact (letters right). Beam alone is weight 0; with the "
          f"language model, weight {best[0]} and bonus {best[1]}. Flips "
          f"count words the language model turns right → wrong and wrong → "
          f"right, against the beam alone. *LM never saw* is the words in "
          f"no training entry of the language model, nor inside one (the "
          f"benchmark words are never entries, but many are inside Norman's "
          f"phrases; cursive_v1's and ManchuAI's are often entries): it "
          f"mustn't help only words it knows.\n")
        p("| set | words | greedy | beam alone | with LM | right → wrong | "
          "wrong → right | LM never saw: words, greedy → with LM |")
        p("|---" * 8 + "|")
        changed = {}
        for name in BENCHMARKS:
            labels, outputs, lengths = data[name]
            greedy = romanized(read_all(pool, outputs, lengths, blank, None,
                                        0), name)
            plain = romanized(read_all(pool, outputs, lengths, blank, 0, 0),
                              name)
            with_lm = romanized(read_all(pool, outputs, lengths, blank,
                                         *best), name)
            cells = [f"{e:.1%} ({c:.1%})" for e, c in
                     (scores(g, labels) for g in (greedy, plain, with_lm))]
            hurt = [(l, a, b) for l, a, b in zip(labels, plain, with_lm)
                    if a == l != b]
            helped = [(l, a, b) for l, a, b in zip(labels, plain, with_lm)
                      if b == l != a]
            changed[name] = (hurt, helped)
            unknown = [i for i, l in enumerate(labels) if l not in seen]
            unknown_cell = (f"{len(unknown)}, " + " → ".join(
                f"{np.mean([g[i] == labels[i] for i in unknown]):.0%}"
                for g in (greedy, with_lm)) if unknown else "0")
            p(f"| {name} | {len(labels)} | " + " | ".join(cells)
              + f" | {len(hurt)} | {len(helped)} | {unknown_cell} |")
            print(f"{name}: " + ", ".join(cells), flush=True)
        p("")
        p("## Examples\n")
        p("true: beam alone → with the language model\n")
        for name, (hurt, helped) in changed.items():
            p(f"### {name}\n")
            p("Made right: " + ", ".join(f"{l}: {a or '∅'} → {b}"
                                         for l, a, b in helped[:15]) + "\n")
            p("Made wrong: " + (", ".join(f"{l}: {a} → {b or '∅'}"
                                          for l, a, b in hurt[:15])
                                or "none") + "\n")

    path = os.path.join(out, "report.md")
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(report))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
