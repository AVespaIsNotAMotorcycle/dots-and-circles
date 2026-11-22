'''
Makes a rendered word look like a photographed woodblock print, so models
trained on it can read real books (see ocr/evaluate_non_synthetic.py). The
result then goes through ocr.preprocess.prepare, exactly as a photo does.

The strengths below are first guesses, checked by eye against the photos.
They deliberately aren't tuned against the photos' scores, so the photos
stay an honest test.
'''

import io
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ocr.preprocess import crop_to_ink, prepare, stretch
from ocr.training_data import WIDTH, fonts, lexicon, pad_top, render

# Stroke thickness, blur and geometry are applied at this width (4 times the
# models' WIDTH), since the full RENDER_SIZE image is ~5 times slower to
# filter. One pixel here is a quarter of a pixel in the models' input.
WORK_WIDTH = 128
# The photos are about this wide before preprocessing, so noise, specks and
# JPEG artifacts are added at this width, at the same scale as theirs.
PHOTO_WIDTH = 64

# Thickening or thinning the strokes by this many WORK_WIDTH pixels, and how
# often. Positive thickens: a MinFilter spreads the dark ink outwards.
STROKE_CHANGES = [-1, 0, 1, 2, 3]
STROKE_WEIGHTS = [0.1, 0.2, 0.3, 0.25, 0.15]
# Chance that only part of the word (one end) gets the stroke change
UNEVEN_STROKE_CHANCE = 0.3
MAX_ROTATION = 3            # degrees either way
# Kept small: shearing a tall word widens its crop, which shrinks the word
# once it's scaled to WIDTH
MAX_SHEAR = 0.07
STRETCH_RANGE = (0.85, 1.2)  # height stretch
# Paper around the word, as a share of its size: photos are never cropped
# tight to the ink
SIDE_MARGIN_RANGE = (0.05, 0.3)
END_MARGIN_RANGE = (0.03, 0.15)
# Chance of a slice of another word at the edge, like neighbouring text
FRAGMENT_CHANCE = 0.25
MAX_BLUR = 1.5              # Gaussian radius at WORK_WIDTH
INK_RANGE = (0, 120)
PAPER_RANGE = (170, 255)
BLOTCH_CHANCE = 0.5
MAX_BLOTCH_DARKNESS = 25
MAX_NOISE = 15              # standard deviation, 0-255 scale
# Kept rare: a speck far out in the margin also widens the crop to the ink
MEAN_SPECKS = 0.4
JPEG_CHANCE = 0.5
JPEG_QUALITY_RANGE = (30, 95)
# Augmenting can shrink a word too much for CTC to fit its letters; it's
# redone this many times with fresh random draws before falling back to the
# clean image
ATTEMPTS = 5
# Pixel rows per model token (as in ocr/train.py)
ROWS_PER_TOKEN = 4

# Bending: the stem weaves sideways along a smooth random curve, as a
# hand's does (augment(..., bend=True)). Run 32 read words bent ±4 model
# pixels about half as well as straight ones (see the diagnostics of
# 2026-11-21). Strengths are, like the rest, first guesses checked by eye
# against the cursive crops, not tuned on any test set's score.
BEND_CHANCE = 0.7
# Largest sideways swing either way, as a share of the word's width (0.08
# of WIDTH is about 4 model pixels)
MAX_BEND = 0.12
# Distance between the curve's control points, as multiples of the word's
# width: about one and a half to four letters
BEND_SPACING_RANGE = (0.6, 1.5)

# Real crops (photos and cursive crops), augmented in training with
# augment_crop: lighter than the rendered words, since they're photos
# already, so no ink levels, noise, specks or JPEG. The same geometry and
# bending, thinner stroke changes, and sometimes the word shrunk inside the
# strip, as when a flourish or slant widens a crop.
CROP_STROKE_CHANGES = [-1, 0, 1, 2]
CROP_STROKE_WEIGHTS = [0.15, 0.45, 0.25, 0.15]
CROP_MAX_BLUR = 1.0         # Gaussian radius at WORK_WIDTH
SHRINK_CHANCE = 0.3
SHRINK_WIDTH_RANGE = (26, 31)   # model pixels, of WIDTH

def scale_to_width(image, width):
    height = max(1, round(image.height * width / image.width))
    return image.resize((width, height), Image.LANCZOS)

def change_strokes(image, change):
    # change > 0 thickens the ink by that many pixels, < 0 thins it
    if change > 0:
        return image.filter(ImageFilter.MinFilter(2 * change + 1))
    if change < 0:
        return image.filter(ImageFilter.MaxFilter(-2 * change + 1))
    return image

def vary_strokes(image, rng):
    '''
    Thickens or thins the strokes, sometimes only towards one end of the
    word, blending smoothly into the rest, as uneven inking does.
    '''
    change = rng.choice(STROKE_CHANGES, p=STROKE_WEIGHTS)
    changed = change_strokes(image, change)
    if change == 0 or rng.random() >= UNEVEN_STROKE_CHANCE:
        return changed
    # A vertical ramp from 0 to 1 across a random band, flipped half the time
    rows = np.linspace(0, 1, image.height)
    start, end = sorted(rng.uniform(0, 1, 2))
    ramp = np.clip((rows - start) / max(end - start, 1e-6), 0, 1)
    if rng.random() < 0.5:
        ramp = 1 - ramp
    mask = Image.fromarray(np.round(np.tile(ramp[:, None], (1, image.width))
                                    * 255).astype(np.uint8))
    return Image.composite(changed, image, mask)

def distort(image, rng):
    # A slight rotation, shear and height stretch, filling with white paper
    height = max(1, round(image.height * rng.uniform(*STRETCH_RANGE)))
    image = image.resize((image.width, height), Image.BICUBIC)
    shear = rng.uniform(-MAX_SHEAR, MAX_SHEAR)
    # Widen first so the sheared word stays inside the image
    extra = int(abs(shear) * image.height) + 1
    wide = Image.new("L", (image.width + 2 * extra, image.height), 255)
    wide.paste(image, (extra, 0))
    image = wide.transform(wide.size, Image.AFFINE,
                           (1, shear, -shear * image.height / 2, 0, 1, 0),
                           resample=Image.BICUBIC, fillcolor=255)
    return image.rotate(rng.uniform(-MAX_ROTATION, MAX_ROTATION),
                        resample=Image.BICUBIC, expand=True, fillcolor=255)

def bend(image, rng):
    '''
    Shifts each row of the word sideways along a smooth random curve, as a
    hand's stem weaves, widening the image so the ink stays inside it.
    '''
    width, height = image.size
    swing = rng.uniform(0, MAX_BEND) * width
    spacing = rng.uniform(*BEND_SPACING_RANGE) * width
    points = max(2, int(np.ceil(height / spacing)) + 1)
    # Control points, smoothly enlarged to one offset per row, as
    # print_and_photograph's shading is
    grid = rng.uniform(-1, 1, (points, 1)).astype(np.float32)
    curve = np.asarray(Image.fromarray(grid, mode="F").resize(
        (1, height), Image.BICUBIC))[:, 0]
    curve = curve / max(1e-6, np.abs(curve).max()) * swing
    room = int(np.ceil(swing)) + 1
    pixels = np.asarray(image, dtype=np.float64)
    columns = np.arange(width + 2 * room)
    source = np.arange(width)
    bent = np.empty((height, width + 2 * room))
    for row in range(height):
        bent[row] = np.interp(columns - room - curve[row], source,
                              pixels[row], left=255, right=255)
    return Image.fromarray(np.round(bent).astype(np.uint8))

def add_margins(image, rng):
    '''
    Surrounds the word with paper, as the photos are, and sometimes puts a
    slice of another word at one edge, like neighbouring text.
    '''
    side = [round(image.width * rng.uniform(*SIDE_MARGIN_RANGE))
            for _ in range(2)]
    ends = [round(image.height * rng.uniform(*END_MARGIN_RANGE))
            for _ in range(2)]
    page = Image.new("L", (image.width + sum(side), image.height + sum(ends)),
                     255)
    page.paste(image, (side[0], ends[0]))

    if rng.random() < FRAGMENT_CHANCE:
        other = render(lexicon[rng.integers(len(lexicon))],
                       fonts[rng.integers(len(fonts))])
        other = scale_to_width(other, image.width)
        # Neighbouring columns sit beside the word, and neighbouring words
        # in the same column above or below it. The slice shows the side of
        # the other word that faces this one, and fills part of the margin.
        edge = rng.choice(["left", "right", "top", "bottom"])
        if edge in ("left", "right"):
            margin = side[0] if edge == "left" else side[1]
            visible = max(1, round(margin * rng.uniform(0.3, 1)))
            height = min(other.height, page.height)
            if edge == "left":
                slice_ = other.crop((other.width - visible, 0,
                                     other.width, height))
                page.paste(slice_, (0, 0))
            else:
                slice_ = other.crop((0, 0, visible, height))
                page.paste(slice_, (page.width - visible, 0))
        else:
            margin = ends[0] if edge == "top" else ends[1]
            visible = max(1, round(margin * rng.uniform(0.3, 1)))
            visible = min(visible, other.height)
            if edge == "top":
                slice_ = other.crop((0, other.height - visible,
                                     other.width, other.height))
                page.paste(slice_, (side[0], 0))
            else:
                slice_ = other.crop((0, 0, other.width, visible))
                page.paste(slice_, (side[0], page.height - visible))
    return page

def print_and_photograph(image, rng):
    '''
    Gives a PHOTO_WIDTH image the look of a photographed page: ink and
    paper levels, uneven shading, noise, specks and sometimes JPEG
    artifacts.
    '''
    pixels = np.asarray(image, dtype=np.float64) / 255
    ink, paper = rng.uniform(*INK_RANGE), rng.uniform(*PAPER_RANGE)
    pixels = ink + (paper - ink) * pixels

    if rng.random() < BLOTCH_CHANCE:
        # Low-frequency shading: a coarse random grid, smoothly enlarged
        grid = rng.uniform(-MAX_BLOTCH_DARKNESS, 0, (rng.integers(2, 6),
                                                     rng.integers(2, 4)))
        grid = Image.fromarray(grid.astype(np.float32), mode="F")
        pixels += np.asarray(grid.resize(image.size, Image.BICUBIC))
    pixels += rng.normal(0, rng.uniform(0, MAX_NOISE), pixels.shape)
    image = Image.fromarray(np.clip(np.round(pixels), 0, 255)
                            .astype(np.uint8))

    draw = ImageDraw.Draw(image)
    for _ in range(rng.poisson(MEAN_SPECKS)):
        x, y = rng.uniform(0, image.width), rng.uniform(0, image.height)
        radius = rng.uniform(0.5, 1.5)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                     fill=int(ink))

    if rng.random() < JPEG_CHANCE:
        buffer = io.BytesIO()
        image.save(buffer, "JPEG",
                   quality=int(rng.integers(*JPEG_QUALITY_RANGE)))
        image = Image.open(buffer).convert("L")
    return image

def tokens_needed(word):
    # CTC needs a token per letter, plus a blank between repeated letters
    repeats = sum(a == b for a, b in zip(word, word[1:]))
    return len(word) + repeats

def augment_once(canvas, rng, bend_words=False):
    return prepare(photograph(canvas, rng, bend_words), crop=True,
                   whiten=True)

def photograph(canvas, rng, bend_words=False):
    '''
    Makes a rendered word look like a photo of a print, before
    preprocessing: greyscale, about PHOTO_WIDTH wide, on grey paper. With
    bend_words, some words are bent too (see bend).
    '''
    image = scale_to_width(canvas, WORK_WIDTH)
    image = vary_strokes(image, rng)
    image = distort(image, rng)
    # Drawn only when bending, so without it every image is as before
    if bend_words and rng.random() < BEND_CHANCE:
        image = bend(image, rng)
    image = add_margins(image, rng)
    blur = rng.uniform(0, MAX_BLUR)
    if blur > 0.2:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    return print_and_photograph(scale_to_width(image, PHOTO_WIDTH), rng)

def augment(canvas, word, seed, epoch, bend_words=False):
    '''
    Turns word, rendered by training_data.render, into a model input that
    looks like a preprocessed photo of a print, bent like handwriting
    sometimes with bend_words. The same seed and epoch always give the
    same image.
    '''
    minimum_height = tokens_needed(word) * ROWS_PER_TOKEN
    for attempt in range(ATTEMPTS):
        image = augment_once(canvas, np.random.default_rng([seed, epoch,
                                                            attempt]),
                             bend_words)
        if image.height >= minimum_height:
            break
    else:
        image = scale_to_width(canvas, WIDTH)
    # The padding varies by epoch too, for more variety in where words sit
    return pad_top(image, random.Random(f"{seed}-{epoch}"))

def shrink(image, rng):
    # The WIDTH-wide word scaled narrower and set somewhere in the strip
    width = int(rng.integers(SHRINK_WIDTH_RANGE[0],
                             SHRINK_WIDTH_RANGE[1] + 1))
    small = scale_to_width(image, width)
    strip = Image.new("L", (WIDTH, small.height), 255)
    strip.paste(small, (int(rng.integers(0, WIDTH - width + 1)), 0))
    return strip

def augment_crop_once(photo, rng):
    '''
    One augmented model input from a real word crop (a greyscale PIL image
    of the photo as it is, before preprocessing).
    '''
    image = crop_to_ink(stretch(photo.convert("L"), whiten=True))
    image = scale_to_width(image, WORK_WIDTH)
    change = rng.choice(CROP_STROKE_CHANGES, p=CROP_STROKE_WEIGHTS)
    image = change_strokes(image, change)
    image = distort(image, rng)
    if rng.random() < BEND_CHANCE:
        image = bend(image, rng)
    blur = rng.uniform(0, CROP_MAX_BLUR)
    if blur > 0.2:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    # Already whitened, so only cropped and scaled to WIDTH now
    image = prepare(image, crop=True, whiten=False)
    if rng.random() < SHRINK_CHANCE:
        image = shrink(image, rng)
    return image

def augment_crop(photo, word, seed):
    '''
    A real word crop augmented for training, as augment does a rendered
    word: seed (a list of ints, e.g. [path's checksum, copy, epoch]) always
    gives the same image. Falls back to the crop prepared as for testing if
    augmenting leaves too few tokens for CTC. Crops aren't padded at the
    top, as the real ones aren't.
    '''
    minimum_height = tokens_needed(word) * ROWS_PER_TOKEN
    for attempt in range(ATTEMPTS):
        image = augment_crop_once(photo,
                                  np.random.default_rng(seed + [attempt]))
        if image.height >= minimum_height:
            return image
    return prepare(photo)
