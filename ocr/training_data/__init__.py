import json
import os
import random

from PIL import Image, ImageChops, ImageDraw, ImageFont

WIDTH = 32
# Tallest an image can be and still fit in a model's context. Padding never
# takes an image past this.
HEIGHT = 256
# Words are rendered at this size, then scaled down to WIDTH.
RENDER_SIZE = 256

directory = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(directory, "lexicon.json"), "r", encoding="utf-8") as file:
    lexicon = json.load(file)

fonts_directory = os.path.join(directory, "fonts")
fonts = sorted(
    os.path.join(fonts_directory, name)
    for name in os.listdir(fonts_directory)
    if name.lower().endswith((".ttf", ".otf"))
)


def render(word, font_path):
    '''
    Renders word at RENDER_SIZE in the font at font_path, cropped to the ink
    and reading top to bottom. Dark ink on white, as a greyscale PIL image.
    '''
    font = ImageFont.truetype(font_path, RENDER_SIZE)

    # Fonts lay Manchu out horizontally, so render it that way first.
    left, top, right, bottom = font.getbbox(word)
    padding = RENDER_SIZE // 4
    canvas = Image.new(
        "L", (right - left + 2 * padding, bottom - top + 2 * padding), 255
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((padding - left, padding - top), word, font=font, fill=0)

    # Crop to the ink, then rotate so the word reads top to bottom.
    background = Image.new("L", canvas.size, 255)
    canvas = canvas.crop(ImageChops.difference(canvas, background).getbbox())
    return canvas.rotate(-90, expand=True)

def pad_top(image, rng):
    '''
    Adds blank rows before the word, so it doesn't always start at the top.
    About a third of words get none, since real images will often be
    cropped right to the ink. rng is a random.Random, so the same rng state
    always gives the same padding.
    '''
    padding = rng.randint(0, max(0, HEIGHT - image.height))
    if rng.random() < 1 / 3:
        padding = 0
    padded = Image.new("L", (WIDTH, image.height + padding), 255)
    padded.paste(image, (0, padding))
    return padded

def generate(seed, epoch=None):
    '''
    Returns (image, word) for a seed. Each integer seed maps to one
    (word, font) pair, so seeds 0 to len(lexicon) * len(fonts) - 1 cover
    every pair exactly once.

    With epoch=None the image is clean, and a seed always gives the same
    one. With an epoch it's augmented to look like a photographed print
    (see augment.py), differently for each epoch but the same for the same
    seed and epoch.
    '''
    if not isinstance(seed, int):
        seed = random.Random(seed).randrange(len(lexicon) * len(fonts))
    word = lexicon[seed % len(lexicon)]
    font_path = fonts[seed // len(lexicon) % len(fonts)]
    canvas = render(word, font_path)

    if epoch is not None:
        # Imported here: augment.py uses ocr.preprocess, which needs this
        # module to have finished loading
        from ocr.training_data.augment import augment
        return augment(canvas, word, seed, epoch), word

    height = max(1, round(canvas.height * WIDTH / canvas.width))
    image = canvas.resize((WIDTH, height), Image.LANCZOS)
    # The padding comes from the seed, so a seed always gives the same image.
    return pad_top(image, random.Random(seed)), word
