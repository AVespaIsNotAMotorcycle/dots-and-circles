'''
Renders synthetic pages of print, for developing and testing the page
reader behind /upload/multiple: Manchu words in columns, read top to bottom
and left to right, sometimes with Chinese characters, along with the true
box, kind and reading order of everything on the page.

A page has one of three layouts:

- "manchu": only Manchu columns.
- "alternating": Manchu and Chinese columns take turns, as in bilingual
  books.
- "mixed": Manchu columns, with short runs of Chinese characters among the
  words, as for glosses and names.

Words ending in a consonant are sometimes followed by the genitive i,
printed apart as a hook, its own item. About a third of pages are set
tight: each word as close to the one before as its ink allows, its tail
running alongside the next word's start, as in much real print.

Pages that aren't alternating may also have small Chinese glosses set in
narrow columns between Manchu ones, as bilingual books print them. Each is
a column of its own, all "other".

Each Manchu word and each Chinese character is one item. An item's box is
the bounding box of its ink, [left, top, right, bottom] with right and
bottom exclusive (as PIL's crop takes it), so neighbouring columns' items
never share a box even when the columns are close.

The Chinese fonts live in fonts/chinese/, which is gitignored like fonts/.
For a .ttc font the traditional Chinese face is used, since that's what
Qing prints use.

Run as python -m ocr.training_data.pages SEED to save a page, the page with
its boxes drawn on, and its items as JSON.
'''

import argparse
import json
import os
import random

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from ocr.photos import manchu_to_abkai
from ocr.split import group_key
from ocr.training_data import HEIGHT, directory, fonts, lexicon

CHINESE_FONTS_DIRECTORY = os.path.join(directory, "fonts", "chinese")
chinese_fonts = sorted(
    os.path.join(CHINESE_FONTS_DIRECTORY, name)
    for name in os.listdir(CHINESE_FONTS_DIRECTORY)
    if name.lower().endswith((".ttf", ".otf", ".ttc"))
) if os.path.isdir(CHINESE_FONTS_DIRECTORY) else []

LAYOUTS = ["manchu", "alternating", "mixed"]
LAYOUT_WEIGHTS = [0.4, 0.3, 0.3]
# What stands in for an item that isn't Manchu in a page's text
OTHER = "□"

# Lengths below are in units of the page's column width: the width of the
# ink of a Manchu column, from its leftmost tooth to its rightmost.
FONT_SIZE_RANGE = (48, 112)        # pixels, for the Manchu font
COLUMN_COUNT_RANGE = (4, 14)
COLUMN_LENGTH_RANGE = (8, 20)      # the height of the text area
WORD_GAP_RANGE = (0.15, 0.5)
CHINESE_GAP_RANGE = (0.05, 0.25)
COLUMN_GAP_RANGE = (0.3, 1.0)
MARGIN_RANGE = (0.5, 2.0)
CHINESE_SIZE_RANGE = (0.8, 1.2)    # a Chinese character's em size
# In a mixed layout, the chance that an item is a run of Chinese
# characters instead of a word, and how long the run can be
CHINESE_RUN_CHANCE = 0.15
CHINESE_RUN_LENGTH = (1, 4)
# Tight setting: the chance a page is set tight, the least clearance
# between two words' ink, up to how much more at random, and the most two
# words may overlap, all in units
TIGHT_CHANCE = 0.35
TIGHT_CLEARANCE = 0.06
TIGHT_SLACK = 0.06
TIGHT_LIMIT = 0.6
# Glosses: the chance a page has them, the chance of a gloss column before
# each Manchu column, their characters' size, how many are in a run, the
# blank between runs (in units), and the gap either side of the column
GLOSS_CHANCE = 0.35
GLOSS_COLUMN_CHANCE = 0.6
GLOSS_SIZE_RANGE = (0.6, 0.95)
GLOSS_RUN_LENGTH = (1, 6)
GLOSS_SPACE_RANGE = (0.5, 6.0)
GLOSS_GAP_RANGE = (0.15, 0.45)
RULES_CHANCE = 0.4                 # a frame and lines between columns
RULE_WIDTH_RANGE = (0.02, 0.06)
# For photo=True
MAX_SKEW = 2.0                     # degrees either way
MAX_BLUR = 0.03

# Common Chinese characters lie in the CJK Unified Ideographs block. Which
# characters they are doesn't matter here, only that they look Chinese.
CJK_RANGE = (0x4E00, 0x9FA5)
COMMON_CHINESE_SHARE = 0.5
INK_THRESHOLD = 128
# The hyphen-like mark the lexicon writes suffixes with (᠊)
NIRUGU = "\u180a"
# The genitive particle i, printed apart from the word before it as a
# hook, and the narrow no-break space that gives it that form. It follows
# a word ending in a consonant GENITIVE_CHANCE of the time, which makes it
# about 4-7% of words (4.5% on the real pages).
GENITIVE = "\u1873"
SUFFIX_SPACE = "\u202f"
GENITIVE_CHANCE = 0.2
VOWELS = set("\u1820\u185d\u1873\u1823\u1860\u1861")
# Words are drawn this far from the edges of their canvas, as a share of the
# font size, so marks that reach past the font's ascent or descent survive
CANVAS_MARGIN = 0.25

_unseen_words = None


def unseen_words():
    '''
    Single words that train.py never trains on: the default for pages, so a
    model can be scored on pages of unseen words. Many lexicon entries are
    phrases, so this is every word of the test entries (as train.py splits
    them) that isn't also a word of a training entry or a training photo.
    Dictionary forms of suffixes, marked with NIRUGU, are left out, since
    print never shows them on their own.
    '''
    global _unseen_words
    if _unseen_words is None:
        # Imported here: augment.py imports this package
        from ocr.photos import manchu_to_roman
        from ocr.split import group_key, load_photo_split, split_words
        from ocr.training_data.augment import ROWS_PER_TOKEN, tokens_needed
        kept = [entry for entry in lexicon
                if tokens_needed(entry) <= HEIGHT // ROWS_PER_TOKEN]
        train, valid, test = split_words(kept)
        trained = {group_key(word) for index in train + valid
                   for word in kept[index].split()}
        train_photos = load_photo_split()["train"]
        _unseen_words = sorted({
            word for index in test for word in kept[index].split()
            if group_key(word) not in trained
            and manchu_to_roman(word) not in train_photos
            and NIRUGU not in word})
    return _unseen_words

def load_chinese_font(path, size):
    # The traditional Chinese face of a collection, or its first
    if not path.lower().endswith(".ttc"):
        return ImageFont.truetype(path, size)
    index = 0
    while True:
        try:
            font = ImageFont.truetype(path, size, index=index)
        except OSError:
            return ImageFont.truetype(path, size)
        if font.getname()[0].endswith(" TC"):
            return font
        index += 1

def render_manchu(word, font):
    '''
    Renders word top to bottom, cropped to its ink along the column but not
    across it, so every word in a font comes out the same width with its
    stem in the same place. Dark ink on white, as a greyscale PIL image.
    '''
    ascent, descent = font.getmetrics()
    margin = round(CANVAS_MARGIN * font.size)
    length = round(font.getlength(word))
    canvas = Image.new("L", (length + 2 * margin,
                             ascent + descent + 2 * margin), 255)
    ImageDraw.Draw(canvas).text((margin, margin), word, font=font, fill=0)
    ink = np.nonzero((np.asarray(canvas) < INK_THRESHOLD).any(axis=0))[0]
    if len(ink):
        canvas = canvas.crop((ink[0], 0, ink[-1] + 1, canvas.height))
    # Fonts lay Manchu out horizontally, top of the letters up: turned
    # clockwise, it reads top to bottom
    return canvas.rotate(-90, expand=True)

def render_chinese(character, font):
    # One character, cropped to its ink
    size = font.size
    canvas = Image.new("L", (2 * size, 2 * size), 255)
    ImageDraw.Draw(canvas).text((size // 2, size // 2), character, font=font,
                                fill=0)
    box = ImageChops.invert(canvas).getbbox()
    return canvas.crop(box) if box else canvas.crop((0, 0, 1, 1))

def ink_columns(image):
    # The first and last (exclusive) pixel columns with ink, or None
    ink = np.nonzero((np.asarray(image) < INK_THRESHOLD).any(axis=0))[0]
    return (int(ink[0]), int(ink[-1]) + 1) if len(ink) else None

_common_chinese = None


def common_chinese():
    '''
    The Chinese characters in the Wiktionary dump, once for each time they
    occur there, so drawing from it favours common characters. Common
    characters are often simple (几, 上, 十), and simple ones look more
    like Manchu than the complex ones that fill most of the Unicode block.
    '''
    global _common_chinese
    if _common_chinese is None:
        path = os.path.join(directory, "kaikki.org-dictionary-Manchu.jsonl")
        with open(path, "r", encoding="utf-8") as file:
            text = file.read()
        _common_chinese = sorted(
            character for character in text
            if CJK_RANGE[0] <= ord(character) <= CJK_RANGE[1])
    return _common_chinese

def chinese_character(rng, font):
    # A random character the font has a glyph for: a common one
    # COMMON_CHINESE_SHARE of the time, otherwise any in the block
    blank = font.getmask("\U0010FFFD").getbbox()
    while True:
        if rng.random() < COMMON_CHINESE_SHARE and common_chinese():
            character = rng.choice(common_chinese())
        else:
            character = chr(rng.randint(*CJK_RANGE))
        box = font.getmask(character).getbbox()
        if box and box != blank:
            return character

def ends_in_consonant(word):
    # Whether a word's last letter, past any invisible marks, is a consonant
    letters = group_key(word)
    return bool(letters) and letters[-1] not in VOWELS

def tightest_gap(above, below, clearance, limit):
    '''
    The most negative gap (overlap), down to -limit pixels, at which below
    can follow above in a column without its ink coming within clearance
    pixels of above's: in tightly set print, a word's tail runs alongside
    the next word's start. Both are rendered words of one font, so they're
    the same width with their stems aligned.
    '''
    ink_above = np.asarray(above) < INK_THRESHOLD
    ink_below = np.asarray(below) < INK_THRESHOLD
    # above's ink, grown by clearance pixels every way
    padded = np.pad(ink_above, clearance)
    grown = np.zeros_like(ink_above)
    height, width_above = ink_above.shape
    for dy in range(2 * clearance + 1):
        for dx in range(2 * clearance + 1):
            grown |= padded[dy:dy + height, dx:dx + width_above]
    width = min(grown.shape[1], ink_below.shape[1])
    for gap in range(-min(limit, above.height - 1, below.height - 1), 0):
        overlap = -gap
        if not (grown[-overlap:, :width] & ink_below[:overlap, :width]).any():
            return gap
    return 0

def fill_column(rng, kind, length, render_word, render_character,
                word_gap, chinese_gap, render_genitive=None, tight=None):
    '''
    Items for one column of the given kind ("manchu", "chinese" or "mixed"),
    as (kind, text, image) tuples, stacked until the next one wouldn't fit
    in length pixels. Returns them with the gap after each. With
    render_genitive, a word ending in a consonant is followed by the
    genitive i GENITIVE_CHANCE of the time. With tight, a callable giving
    the gap between two rendered words, Manchu following Manchu is set that
    close instead.
    '''
    items = []
    used = 0
    failures = 0
    while failures < 3:
        if kind == "chinese" or (kind == "mixed"
                                 and rng.random() < CHINESE_RUN_CHANCE):
            run = [render_character()
                   for _ in range(1 if kind == "chinese"
                                  else rng.randint(*CHINESE_RUN_LENGTH))]
            gaps = [chinese_gap()] * (len(run) - 1) + [word_gap()]
            new = [("other",) + item + (gap,)
                   for item, gap in zip(run, gaps)]
        else:
            word, image = render_word()
            new = [("manchu", word, image, word_gap())]
            if (render_genitive and ends_in_consonant(word)
                    and rng.random() < GENITIVE_CHANCE):
                new.append(("manchu", GENITIVE, render_genitive(),
                            word_gap()))
            if tight:
                # Each word as close to the one before as its ink allows
                for index in range(len(new) - 1):
                    new[index] = new[index][:3] + (tight(new[index][2],
                                                         new[index + 1][2]),)
        # The gap after the item before, if this one is set tight against it
        change = 0
        if tight and new[0][0] == "manchu" and items \
                and items[-1][0] == "manchu":
            closer = tight(items[-1][2], new[0][2])
            change = closer - items[-1][3]
        height = sum(image.height + gap for _, _, image, gap in new)
        if used + change + height - new[-1][3] > length:
            failures += 1
            continue
        if change:
            items[-1] = items[-1][:3] + (items[-1][3] + change,)
        items.extend(new)
        used += change + height
    return items

def fill_gloss(rng, length, unit, render_gloss, chinese_gap):
    '''
    A column of small Chinese glosses, set between two Manchu columns: runs
    of characters with blank stretches between, as (offset, items, True),
    where offset is the blank above the first run and items are as
    fill_column gives them.
    '''
    items = []
    offset = round(unit * rng.uniform(0, GLOSS_SPACE_RANGE[1] / 2))
    used = offset
    while True:
        run = [render_gloss() for _ in range(rng.randint(*GLOSS_RUN_LENGTH))]
        space = round(unit * rng.uniform(*GLOSS_SPACE_RANGE))
        gaps = [chinese_gap() for _ in run[:-1]] + [space]
        height = sum(image.height for _, image in run) + sum(gaps[:-1])
        if used + height > length:
            break
        items.extend(("other", text, image, gap)
                     for (text, image), gap in zip(run, gaps))
        used += height + space
    if not items:
        text, image = render_gloss()
        items = [("other", text, image, 0)]
    return offset, items, True

def generate_page(seed, layout=None, photo=False, rules=None, words=None,
                  glosses=None, tight=None):
    '''
    Returns (image, items) for a seed: a greyscale PIL page, and its items
    in reading order. Each item is a dict with:

    - "kind": "manchu" or "other"
    - "text": the Manchu word or the Chinese character
    - "box": [left, top, right, bottom] of its ink, right and bottom
      exclusive
    - "column": its column, counting from 0 at the left
    - "position": its place in the column, counting from 0 at the top

    The same arguments always give the same page. layout is one of LAYOUTS
    (by default, chosen by the seed). photo=True skews, blurs and shades the
    page like a photograph. rules says whether to draw a frame and lines
    between columns (by default, chosen by the seed). words are the Manchu
    words to use (by default, unseen_words()). glosses says whether to set
    small Chinese glosses between some Manchu columns (by default, chosen
    by the seed, never on alternating pages). tight says whether to set
    Manchu words as close as their ink allows, a word's tail running
    alongside the next word's start (by default, chosen by the seed).
    '''
    if not fonts:
        raise RuntimeError("No Manchu fonts in training_data/fonts/.")
    if not chinese_fonts:
        raise RuntimeError(
            f"No Chinese fonts in {CHINESE_FONTS_DIRECTORY}.")
    rng = random.Random(seed)
    words = words if words is not None else unseen_words()
    layout = layout or rng.choices(LAYOUTS, LAYOUT_WEIGHTS)[0]
    rules = rng.random() < RULES_CHANCE if rules is None else rules

    # One typeface per page, as a book has
    font = ImageFont.truetype(rng.choice(fonts),
                              rng.randint(*FONT_SIZE_RANGE))
    sample = render_manchu("ᠠᠪᡴᠠ ᠨᠠ", font)
    unit = max(1, ink_columns(sample)[1] - ink_columns(sample)[0])
    chinese_font = load_chinese_font(
        rng.choice(chinese_fonts),
        max(8, round(unit * rng.uniform(*CHINESE_SIZE_RANGE))))

    def scaled(bounds):
        return max(1, round(unit * rng.uniform(*bounds)))
    word_gap_base = rng.uniform(*WORD_GAP_RANGE)
    chinese_gap_base = rng.uniform(*CHINESE_GAP_RANGE)
    # Gaps vary a little around the page's own spacing
    word_gap = lambda: max(1, round(unit * word_gap_base
                                    * rng.uniform(0.7, 1.3)))
    chinese_gap = lambda: max(1, round(unit * chinese_gap_base
                                       * rng.uniform(0.7, 1.3)))
    def render_word():
        word = rng.choice(words)
        return word, render_manchu(word, font)
    def render_character():
        character = chinese_character(rng, chinese_font)
        return character, render_chinese(character, chinese_font)
    # After U+202F the font draws the genitive i in its suffix form, the
    # hook print uses, not the isolated form a lone i gets
    genitive_image = render_manchu(SUFFIX_SPACE + GENITIVE, font)
    render_genitive = lambda: genitive_image

    column_count = rng.randint(*COLUMN_COUNT_RANGE)
    length = scaled(COLUMN_LENGTH_RANGE)
    chinese_first = rng.random() < 0.5
    glossed = glosses if glosses is not None else (
        layout != "alternating" and rng.random() < GLOSS_CHANCE)
    gloss_font = load_chinese_font(
        rng.choice(chinese_fonts),
        max(8, round(unit * rng.uniform(*GLOSS_SIZE_RANGE))))
    def render_gloss():
        character = chinese_character(rng, gloss_font)
        return character, render_chinese(character, gloss_font)

    tight = rng.random() < TIGHT_CHANCE if tight is None else tight
    clearance = max(1, round(TIGHT_CLEARANCE * unit))
    slack = max(0, round(TIGHT_SLACK * unit))
    def set_tight(above, below):
        return (tightest_gap(above, below, clearance,
                             round(TIGHT_LIMIT * unit))
                + rng.randint(0, slack))

    # Each column is (offset from the top, items, whether it's a gloss)
    columns = []
    for index in range(column_count):
        if layout == "alternating":
            kind = "chinese" if (index % 2 == 0) == chinese_first else "manchu"
        else:
            kind = layout if layout == "mixed" else "manchu"
        if (glossed and index > 0 and kind != "chinese"
                and rng.random() < GLOSS_COLUMN_CHANCE):
            columns.append(fill_gloss(rng, length, unit, render_gloss,
                                      chinese_gap))
        columns.append((0, fill_column(rng, kind, length, render_word,
                                       render_character, word_gap,
                                       chinese_gap, render_genitive,
                                       set_tight if tight else None), False))

    # Each column's ink spans the same width around the stem for every
    # word, so its width is the widest word's ink. Chinese characters are
    # centred on that span.
    spans = []
    for _, items, _ in columns:
        manchu = [ink_columns(image) for kind, _, image, _ in items
                  if kind == "manchu" and ink_columns(image)]
        if manchu:
            spans.append((min(a for a, _ in manchu),
                          max(b for _, b in manchu)))
        else:
            width = max((image.width for _, _, image, _ in items), default=1)
            spans.append((0, width))

    margin = scaled(MARGIN_RANGE)
    gaps = [scaled(GLOSS_GAP_RANGE if left[2] or right[2]
                   else COLUMN_GAP_RANGE)
            for left, right in zip(columns, columns[1:])]
    width = (2 * margin + sum(b - a for a, b in spans) + sum(gaps))
    height = 2 * margin + length
    page = Image.new("L", (width, height), 255)
    # Each item's ink, by its index + 1, to find its box after any skew
    owners = Image.new("I", (width, height), 0)

    items = []
    lefts = []
    x = margin
    for column, ((offset, column_items, _), (start, end)) in enumerate(
            zip(columns, spans)):
        lefts.append(x)
        y = margin + offset
        for position, (kind, text, image, gap) in enumerate(column_items):
            if kind == "manchu":
                left = x - start
            else:
                left = x + (end - start - image.width) // 2
            # Masked by its own darkness, so its white leaves the page be
            page.paste(image, (left, y), ImageChops.invert(image))
            mask = Image.eval(image, lambda value: 255
                              if value < INK_THRESHOLD else 0)
            owners.paste(len(items) + 1, (left, y, left + image.width,
                                          y + image.height), mask)
            items.append({"kind": kind, "text": text, "column": column,
                          "position": position})
            y += image.height + gap
        x += end - start + (gaps[column] if column < len(gaps) else 0)

    if rules:
        draw = ImageDraw.Draw(page)
        rule = scaled(RULE_WIDTH_RANGE)
        inset = margin // 2
        draw.rectangle((inset, inset, width - 1 - inset, height - 1 - inset),
                       outline=0, width=rule)
        for column in range(1, len(columns)):
            middle = (lefts[column - 1] + spans[column - 1][1]
                      - spans[column - 1][0] + lefts[column]) // 2
            draw.line((middle, inset, middle, height - 1 - inset), fill=0,
                      width=rule)

    if photo:
        skew = rng.uniform(-MAX_SKEW, MAX_SKEW)
        page = page.rotate(skew, resample=Image.BICUBIC, expand=True,
                           fillcolor=255)
        owners = owners.rotate(skew, resample=Image.NEAREST, expand=True,
                               fillcolor=0)
        blur = rng.uniform(0, MAX_BLUR) * unit
        if blur > 0.3:
            page = page.filter(ImageFilter.GaussianBlur(blur))
        # Imported here: augment.py imports this package
        from ocr.training_data.augment import print_and_photograph
        page = print_and_photograph(page, np.random.default_rng(
            rng.randrange(2**32)))

    for item, box in zip(items, ink_boxes(owners, len(items))):
        item["box"] = box
    return page, items

def ink_boxes(owners, count):
    # The ink box of each of count items in an owners image, or None
    ids = np.asarray(owners)
    ys, xs = np.nonzero(ids)
    found = ids[ys, xs] - 1
    lefts = np.full(count, np.iinfo(np.int64).max)
    tops = lefts.copy()
    rights = np.full(count, -1)
    bottoms = rights.copy()
    np.minimum.at(lefts, found, xs)
    np.minimum.at(tops, found, ys)
    np.maximum.at(rights, found, xs + 1)
    np.maximum.at(bottoms, found, ys + 1)
    return [None if rights[index] < 0 else
            [int(lefts[index]), int(tops[index]), int(rights[index]),
             int(bottoms[index])] for index in range(count)]

def page_lines(items, abkai=False):
    '''
    The page's text as /upload/multiple returns it: one line per column with
    any Manchu, words separated by spaces, with OTHER for each item that
    isn't Manchu. abkai=True writes the words in Abkai.
    '''
    columns = {}
    for item in items:
        columns.setdefault(item["column"], []).append(item)
    lines = []
    for column in sorted(columns):
        column_items = sorted(columns[column], key=lambda item:
                              item["position"])
        if all(item["kind"] != "manchu" for item in column_items):
            continue
        # A Manchu box read as nothing (a stray mark) is left out
        lines.append(" ".join(
            OTHER if item["kind"] != "manchu"
            else manchu_to_abkai(item["text"]) if abkai else item["text"]
            for item in column_items
            if item["kind"] != "manchu" or item["text"]))
    return "\n".join(lines)

def draw_boxes(page, items):
    # The page in colour with Manchu boxes green and others red
    image = page.convert("RGB")
    draw = ImageDraw.Draw(image)
    for item in items:
        if item["box"]:
            left, top, right, bottom = item["box"]
            draw.rectangle((left, top, right - 1, bottom - 1),
                           outline=(0, 160, 0) if item["kind"] == "manchu"
                           else (220, 0, 0), width=2)
    return image

def main():
    parser = argparse.ArgumentParser(
        description="Render a synthetic page and its boxes.")
    parser.add_argument("seed", type=int)
    parser.add_argument("--layout", choices=LAYOUTS)
    parser.add_argument("--photo", action="store_true",
                        help="make it look like a photograph")
    parser.add_argument("-o", "--output", default=".",
                        help="folder to save to (default: here)")
    args = parser.parse_args()
    page, items = generate_page(args.seed, args.layout, args.photo)
    stem = os.path.join(args.output, f"page{args.seed}")
    page.save(f"{stem}.png")
    draw_boxes(page, items).save(f"{stem}_boxes.png")
    with open(f"{stem}.json", "w", encoding="utf-8") as file:
        json.dump({"items": items, "manchu": page_lines(items),
                   "abkai": page_lines(items, abkai=True)}, file,
                  ensure_ascii=False, indent=1)
    print(f"Saved {stem}.png, {stem}_boxes.png and {stem}.json.")

if __name__ == "__main__":
    main()
