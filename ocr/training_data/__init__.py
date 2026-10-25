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


def generate(seed):
    # Each integer seed maps to one (word, font) pair, so seeds
    # 0 to len(lexicon) * len(fonts) - 1 cover every pair exactly once.
    if not isinstance(seed, int):
        seed = random.Random(seed).randrange(len(lexicon) * len(fonts))
    word = lexicon[seed % len(lexicon)]
    font = ImageFont.truetype(
        fonts[seed // len(lexicon) % len(fonts)], RENDER_SIZE
    )

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
    canvas = canvas.rotate(-90, expand=True)

    height = max(1, round(canvas.height * WIDTH / canvas.width))
    image = canvas.resize((WIDTH, height), Image.LANCZOS)

    # Add blank rows before the word, so it doesn't always start at the top.
    # The amount comes from the seed, so a seed always gives the same image.
    padding = random.Random(seed).randint(0, max(0, HEIGHT - height))
    padded = Image.new("L", (WIDTH, height + padding), 255)
    padded.paste(image, (0, padding))
    return padded, word
