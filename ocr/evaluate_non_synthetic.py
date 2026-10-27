# Run from the repository root with:
#   .venv/bin/python -m ocr.evaluate_non_synthetic ocr/runs/run025/weights_D-8-64_Adam_3e-3.npz

import os
# Each worker process runs its own model, so numpy's BLAS shouldn't start
# threads of its own too. This only works if it's set before numpy is
# first imported.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import csv
import random
import time
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr.modelD import ModelD
from ocr.photos import (DATA_DIRECTORY, load_photo, manchu_to_abkai,
                        sample_images)
from ocr.train import (CONTEXT_LENGTH, END, NUM_CLASSES, TOKEN_DIM,
                       edit_distance, image_to_array, predict, read_words,
                       token_count)
from ocr.training_data import fonts, render

# Images per batch. The models keep every layer's activations from the
# forward pass, so memory grows quickly with batch size; 16 images with 8
# workers stays around 2 GB.
BATCH_SIZE = 16
WORKERS = 8
# Example images show each word at this height, the model's guess rendered
# in EXAMPLE_FONT and captioned in CAPTION_FONT (PIL's own font if missing)
EXAMPLE_HEIGHT = 300
EXAMPLE_FONT = next(path for path in fonts if "ZhengHei" in path)
CAPTION_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
# The preprocessing each pass tests: (crop to ink, whiten the paper)
PASSES = {
    "plain": (False, False),
    "cropped": (True, False),
    "whitened": (False, True),
    "whitened_cropped": (True, True),
}

# Each worker process loads the model once, in init_worker
model = None

def init_worker(weights, depth, emb_dim):
    global model
    model = ModelD(token_dim=TOKEN_DIM, context_length=CONTEXT_LENGTH,
                   num_classes=NUM_CLASSES, depth=depth, blank=END,
                   emb_dim=emb_dim)
    model.load_weights(weights)
    model.set_training(False)

def read_batch(job):
    '''
    Reads one batch of (path, label) pairs, returning a row of results for
    each image.
    '''
    batch, crop, whiten = job
    images = [load_photo(path, crop, whiten) for path, _ in batch]
    entries = [(image_to_array(image), None, token_count(image))
               for image in images]
    _, y = predict(model, entries)
    rows = []
    for (path, label), image, guess in zip(batch, images,
                                           read_words(model, y, entries)):
        abkai = manchu_to_abkai(guess)
        distance = edit_distance(abkai, label)
        rows.append({"path": os.path.relpath(path, DATA_DIRECTORY),
                     "label": label, "guess": guess, "guess_abkai": abkai,
                     "height": image.height, "edit_distance": distance,
                     "char_accuracy": max(0, 1 - distance / len(label)),
                     "exact": int(abkai == label)})
    return rows

def evaluate(pool, sample, crop, whiten, deadline):
    '''
    Reads every image in sample, stopping early if time.time() passes
    deadline. Returns the rows of the images that were read.
    '''
    jobs = [(sample[start:start + BATCH_SIZE], crop, whiten)
            for start in range(0, len(sample), BATCH_SIZE)]
    rows = []
    for batch_rows in pool.imap(read_batch, jobs):
        rows.extend(batch_rows)
        if time.time() > deadline:
            print(f"Time limit reached after {len(rows)} images.")
            break
    return rows

def summarize(name, rows):
    exact = sum(row["exact"] for row in rows)
    char_accuracy = np.mean([row["char_accuracy"] for row in rows])
    print(f"{name}: {exact} of {len(rows)} words exact "
          f"({exact / len(rows):.1%}), mean char accuracy "
          f"{char_accuracy:.1%}")

def render_word(word):
    # The guess as a lexigraph, or None for an empty guess
    if not word.strip():
        return None
    return render(word, EXAMPLE_FONT)

def caption_font(size):
    try:
        return ImageFont.truetype(CAPTION_FONT, size)
    except OSError:
        return ImageFont.load_default()

def save_examples(rows, count, out, seed):
    '''
    Saves count random rows as images: the photo as it was before
    preprocessing on the left, the model's guess rendered as a lexigraph on
    the right, with the Abkai under each.
    '''
    column = 220
    title_font, word_font = caption_font(14), caption_font(18)
    picks = random.Random(seed).sample(rows, min(count, len(rows)))
    for number, row in enumerate(picks, start=1):
        photo = Image.open(os.path.join(DATA_DIRECTORY, row["path"]))
        guess = render_word(row["guess"])
        sheet = Image.new("RGB", (2 * column, EXAMPLE_HEIGHT + 110), "white")
        draw = ImageDraw.Draw(sheet)
        sides = [(photo, "Photo (label)", row["label"]),
                 (guess, "Model's guess", row["guess_abkai"] or "(empty)")]
        for side, (image, title, abkai) in enumerate(sides):
            middle = side * column + column // 2
            if image is not None:
                width = max(1, round(image.width * EXAMPLE_HEIGHT
                                     / image.height))
                image = image.convert("RGB").resize(
                    (width, EXAMPLE_HEIGHT), Image.LANCZOS)
                sheet.paste(image, (middle - width // 2, 16))
            draw.text((middle, EXAMPLE_HEIGHT + 32), title, font=title_font,
                      fill="#555555", anchor="mm")
            draw.text((middle, EXAMPLE_HEIGHT + 62), abkai, font=word_font,
                      fill="black", anchor="mm")
        if row["exact"]:
            verdict, colour = "exact match", "#2a78d6"
        else:
            verdict, colour = f"{row['edit_distance']} edit(s) off", "#555555"
        draw.text((column, EXAMPLE_HEIGHT + 95), verdict, font=title_font,
                  fill=colour, anchor="mm")
        path = os.path.join(out, f"non_synthetic_example_{number}.png")
        sheet.save(path)
        print(f"Wrote {path}: {row['label']} -> "
              f"{row['guess_abkai'] or '(empty)'}")

def main():
    parser = argparse.ArgumentParser(
        description="Tests a ModelD's weights on photographed Manchu words.")
    parser.add_argument("weights", help="a weights .npz file")
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--emb-dim", type=int, default=64)
    parser.add_argument("--sample", type=int, default=5000,
                        help="number of images to test")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minutes", type=float, default=5,
                        help="time limit for the whole test")
    parser.add_argument("--passes", nargs="+", choices=list(PASSES),
                        default=list(PASSES),
                        help="which preprocessing to test, each on the "
                        "same sample (default: all of them)")
    parser.add_argument("--out", help="folder to write the results CSVs to "
                        "(default: the weights' folder)")
    parser.add_argument("--examples", type=int, default=0,
                        help="also save this many random example images, "
                        "photo beside the model's guess, from the last pass")
    args = parser.parse_args()

    deadline = time.time() + args.minutes * 60
    sample = sample_images(args.sample, args.seed)
    out = args.out or os.path.dirname(os.path.abspath(args.weights))
    rows = []

    with Pool(WORKERS, initializer=init_worker,
              initargs=(args.weights, args.depth, args.emb_dim)) as pool:
        for name in args.passes:
            crop, whiten = PASSES[name]
            if time.time() > deadline:
                print(f"No time left for the {name} pass.")
                break
            rows = evaluate(pool, sample, crop, whiten, deadline)
            summarize(name, rows)
            path = os.path.join(out, f"non_synthetic_eval_{name}.csv")
            with open(path, "w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            print(f"Wrote {path}")

    if args.examples and rows:
        save_examples(rows, args.examples, out, args.seed)

if __name__ == "__main__":
    main()
