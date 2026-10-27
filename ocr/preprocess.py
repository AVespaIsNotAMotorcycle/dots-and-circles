'''
Prepares a greyscale word image for the models the same way whether it's a
photographed word or an augmented synthetic one: contrast stretched, cropped
to the ink, and scaled to WIDTH pixels wide.
'''

import numpy as np
from PIL import Image

from ocr.training_data import WIDTH

# Pixels darker than this, after the contrast stretch, count as ink when
# cropping
INK_THRESHOLD = 128
# When whitening, the darkest INK_PERCENTILE % of pixels become black and the
# lightest (100 - PAPER_PERCENTILE) % white. Ink rarely covers more than a
# fifth of a word's photo, so the PAPER_PERCENTILE pixel is paper.
INK_PERCENTILE = 2
PAPER_PERCENTILE = 70

def stretch(image, whiten=False):
    '''
    Stretches a greyscale image's contrast so its darkest pixel is black and
    its lightest white. whiten=True stretches from percentiles instead:
    INK_PERCENTILE becomes black and PAPER_PERCENTILE white, with everything
    beyond clipped, so grey paper turns white like the training images'
    background.
    '''
    pixels = np.asarray(image, dtype=np.float64)
    if whiten:
        darkest, lightest = np.percentile(pixels, [INK_PERCENTILE,
                                                   PAPER_PERCENTILE])
    else:
        darkest, lightest = pixels.min(), pixels.max()
    if lightest > darkest:
        pixels = (pixels - darkest) / (lightest - darkest) * 255
    pixels = np.clip(pixels, 0, 255)
    return Image.fromarray(np.round(pixels).astype(np.uint8))

def crop_to_ink(image):
    # Crops to the pixels darker than INK_THRESHOLD, if there are any
    ink = np.asarray(image) < INK_THRESHOLD
    rows = np.nonzero(ink.any(axis=1))[0]
    columns = np.nonzero(ink.any(axis=0))[0]
    if len(rows) and len(columns):
        image = image.crop((columns[0], rows[0],
                            columns[-1] + 1, rows[-1] + 1))
    return image

def prepare(image, crop=True, whiten=True):
    '''
    Stretches, optionally crops, and scales a word image to WIDTH pixels
    wide, keeping its proportions. Takes and returns greyscale PIL images,
    dark ink on light paper.
    '''
    image = stretch(image.convert("L"), whiten)
    if crop:
        image = crop_to_ink(image)
    height = max(1, round(image.height * WIDTH / image.width))
    return image.resize((WIDTH, height), Image.LANCZOS)
