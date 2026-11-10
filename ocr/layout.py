'''
Finds the words on a photographed or scanned page of vertical print, for
the page reader behind /upload/multiple. Classical image processing, with no
learned parts:

1. Binarize: divide out the paper's shading, then split ink from paper with
   Otsu's threshold.
2. Deskew: try small rotations and keep the one whose vertical projection
   (ink per pixel column) has the sharpest gutters between columns.
3. Remove ruled lines: long straight runs of ink that nothing touches
   along their length, as a frame or the lines between columns are. A
   Manchu stem is as long and straight, but its teeth touch it.
4. Columns: the stretches of pixel columns with ink, split where two
   columns touch.
5. Words: within each column, the stretches of rows with ink, joined across
   gaps too small to be between words.

Boxes are in the deskewed image, which find_words returns too, so draw them
on that. Page.to_input maps a box back to the image as given.

Lengths are measured in the page's unit: the median width of a column's
ink, from its leftmost tooth to its rightmost.
'''

import math

import numpy as np
from PIL import Image, ImageFilter

# Binarizing: the paper is estimated at 1/BACKGROUND_SCALE of the size,
# where a BACKGROUND_FILTER-pixel maximum filter covers the ink
BACKGROUND_SCALE = 8
BACKGROUND_FILTER = 5
# Deskewing, on a copy at most DESKEW_SIZE pixels on its longest side
DESKEW_SIZE = 600
MAX_SKEW = 5.0              # degrees either way
COARSE_STEP = 0.25
FINE_STEP = 0.05
# Ruled lines: runs at least this share of the ink's extent, touched by
# other ink along at most RULE_CONTACT of their length
RULE_LENGTH = 0.6
RULE_CONTACT = 0.1
# Breaks in a line up to this share of the ink's extent are bridged, and up
# to RULE_EDGE_ROWS rows either side inked along at least RULE_EDGE of the
# line are its blurred edges
RULE_GAP = 0.01
RULE_EDGE = 0.5
RULE_EDGE_ROWS = 2
# Binarizing: ink pixels with fewer inked neighbours than this are noise
SPECK_NEIGHBOURS = 2
# Columns: the unit is measured on stretches of ink at least this share as
# wide as the widest
UNIT_SHARE = 0.3
# Columns: pixel columns with at most this share of the tallest column's
# ink count as gutter
GUTTER_INK = 0.01
# A stretch of ink wider than this many units is two columns touching
SPLIT_WIDTH = 1.6
# Stretches narrower than this many units are noise or stray marks, unless
# they're close to a column
MIN_COLUMN_WIDTH = 0.25
# Words: gaps of at least WORD_GAP units separate words; pieces shorter
# than FRAGMENT units join their nearer neighbour, as do pieces with less
# than MIN_INK square units of ink
WORD_GAP = 0.12
FRAGMENT = 0.2
MIN_INK = 0.03


class Page:
    '''
    The result of find_words:

    - image: the deskewed greyscale page, a PIL image
    - ink: a boolean array of its ink, without ruled lines
    - angle: the rotation, in degrees anticlockwise, that deskewed it
    - unit: the page's unit, in pixels
    - words: dicts with "box" ([left, top, right, bottom], right and bottom
      exclusive), "column" (from 0 at the left) and "position" (from 0 at
      the top of the column), in reading order
    - input_size: the size of the image as given
    '''

    def __init__(self, image, ink, angle, unit, words, input_size):
        self.image = image
        self.ink = ink
        self.angle = angle
        self.unit = unit
        self.words = words
        self.input_size = input_size

    def to_input(self, box):
        '''
        The box of the image as given that holds box, a box of the
        deskewed image.
        '''
        left, top, right, bottom = box
        corners = [unrotate_point(x, y, self.angle, self.input_size,
                                  self.image.size)
                   for x, y in ((left, top), (right, top), (left, bottom),
                                (right, bottom))]
        xs, ys = zip(*corners)
        width, height = self.input_size
        return [max(0, math.floor(min(xs))), max(0, math.floor(min(ys))),
                min(width, math.ceil(max(xs))),
                min(height, math.ceil(max(ys)))]

def unrotate_point(x, y, angle, input_size, output_size):
    '''
    Where a point of image.rotate(angle, expand=True) was in image, given
    both sizes. PIL rotates anticlockwise about the centre, in coordinates
    where y grows downward.
    '''
    radians = math.radians(angle)
    dx, dy = x - output_size[0] / 2, y - output_size[1] / 2
    return (input_size[0] / 2 + dx * math.cos(radians)
            - dy * math.sin(radians),
            input_size[1] / 2 + dx * math.sin(radians)
            + dy * math.cos(radians))

def binarize(image):
    '''
    The ink of a greyscale PIL image, as a boolean array: the image divided
    by an estimate of the bare paper, thresholded by Otsu's method.
    '''
    small = image.resize((max(1, image.width // BACKGROUND_SCALE),
                          max(1, image.height // BACKGROUND_SCALE)),
                         Image.BOX)
    paper = (small.filter(ImageFilter.MaxFilter(BACKGROUND_FILTER))
             .filter(ImageFilter.GaussianBlur(BACKGROUND_FILTER / 2))
             .resize(image.size, Image.BILINEAR))
    pixels = np.asarray(image, dtype=np.float64)
    paper = np.maximum(np.asarray(paper, dtype=np.float64), 1)
    levels = np.clip(np.round(pixels / paper * 255), 0, 255).astype(int)
    return despeckle(levels < otsu(levels))

def despeckle(ink):
    # The ink without pixels that have fewer than SPECK_NEIGHBOURS of their
    # 8 neighbours inked: noise, rather than strokes
    padded = np.pad(ink, 1).astype(np.int8)
    neighbours = sum(padded[1 + dy:padded.shape[0] - 1 + dy,
                            1 + dx:padded.shape[1] - 1 + dx]
                     for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                     if dy or dx)
    return ink & (neighbours >= SPECK_NEIGHBOURS)

def otsu(levels):
    # The threshold (0-256) that best separates a histogram of 0-255 levels
    counts = np.bincount(levels.ravel(), minlength=256).astype(np.float64)
    total = counts.sum()
    below = np.cumsum(counts)
    below_sum = np.cumsum(counts * np.arange(256))
    above = total - below
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_below = below_sum / below
        mean_above = (below_sum[-1] - below_sum) / above
        between = below * above * (mean_below - mean_above) ** 2
    between = np.nan_to_num(between)
    return int(np.argmax(between)) + 1

def sharpness(xs, ys, weights, angle):
    '''
    How sharply the vertical projection of ink at (xs, ys), with weights,
    changes once the page is rotated by angle degrees anticlockwise: high
    when its columns stand straight. The points are projected directly,
    rather than the image rotated, so no angle gains or loses sharpness
    from resampling.
    '''
    radians = math.radians(angle)
    # A point's x once rotated anticlockwise (y grows downward)
    positions = xs * math.cos(radians) + ys * math.sin(radians)
    profile = np.bincount(np.floor(positions - positions.min()).astype(int),
                          weights=weights)
    return float(np.sum(np.diff(profile) ** 2))

def best_angle(angles, xs, ys, weights):
    # The angle with the sharpest projection, or the middle of those tied
    scores = np.array([sharpness(xs, ys, weights, angle) for angle in angles])
    tied = np.nonzero(scores >= scores.max() * (1 - 1e-9))[0]
    return float(angles[tied[len(tied) // 2]])

def find_skew(ink):
    '''
    The rotation, in degrees, that best stands the ink's columns upright:
    found coarsely on a copy at most DESKEW_SIZE pixels on its longest side,
    then finely at full size, where a small turn still moves the ink.
    '''
    scale = min(1, DESKEW_SIZE / max(ink.shape))
    small = np.asarray(Image.fromarray(ink.astype(np.uint8) * 255).resize(
        (max(1, round(ink.shape[1] * scale)),
         max(1, round(ink.shape[0] * scale))), Image.BOX), dtype=np.float64)
    ys, xs = np.nonzero(small)
    if not len(xs):
        return 0.0
    angles = np.arange(-MAX_SKEW, MAX_SKEW + COARSE_STEP / 2, COARSE_STEP)
    coarse = best_angle(angles, xs, ys, small[ys, xs])
    ys, xs = np.nonzero(ink)
    angles = np.arange(coarse - COARSE_STEP,
                       coarse + COARSE_STEP + FINE_STEP / 2, FINE_STEP)
    return best_angle(angles, xs.astype(np.float64), ys.astype(np.float64),
                      None)

def runs(line_mask):
    '''
    The runs of True along each row of a 2D boolean array, as arrays
    (rows, starts, ends), ends exclusive.
    '''
    padded = np.pad(line_mask, ((0, 0), (1, 1))).astype(np.int8)
    change = np.diff(padded, axis=1)
    rows, starts = np.nonzero(change == 1)
    _, ends = np.nonzero(change == -1)
    return rows, starts, ends

def close_gaps(mask, length):
    '''
    The mask with gaps of at most length pixels along its rows filled in,
    where there's ink on both sides.
    '''
    rows, starts, ends = runs(~mask)
    interior = (starts > 0) & (ends < mask.shape[1]) & (ends - starts
                                                        <= length)
    change = np.zeros((mask.shape[0], mask.shape[1] + 1), dtype=np.int32)
    np.add.at(change, (rows[interior], starts[interior]), 1)
    np.add.at(change, (rows[interior], ends[interior]), -1)
    return mask | (np.cumsum(change, axis=1)[:, :-1] > 0)

def remove_rules(ink):
    '''
    The ink without ruled lines: horizontal or vertical runs at least
    RULE_LENGTH of the ink's extent, with other ink touching their sides
    along at most RULE_CONTACT of their length. Small breaks in a line, from
    blur and noise, are bridged first. A line is several pixels thick, so
    neighbouring runs are taken together, along with the blurred rows
    either side.
    '''
    ink = ink.copy()
    rows = np.nonzero(ink.any(axis=1))[0]
    columns = np.nonzero(ink.any(axis=0))[0]
    if not len(rows):
        return ink
    for transposed in (False, True):
        mask = ink.T if transposed else ink
        # Runs go along the mask's rows: across the page, then down it
        extent = (rows[-1] - rows[0] if transposed
                  else columns[-1] - columns[0])
        closed = close_gaps(mask, max(2, round(RULE_GAP * extent)))
        line_rows, starts, ends = runs(closed)
        long = ends - starts >= RULE_LENGTH * extent
        if not long.any():
            continue
        long_rows = np.zeros(mask.shape[0], dtype=bool)
        long_rows[line_rows[long]] = True
        for first, last in stretches(long_rows):
            selected = long & (line_rows >= first) & (line_rows < last)
            start = starts[selected].min()
            end = ends[selected].max()
            # Take in the blurred edges: up to RULE_EDGE_ROWS rows either
            # side with ink along at least RULE_EDGE of the line
            for _ in range(RULE_EDGE_ROWS):
                if (first > 0 and closed[first - 1, start:end].mean()
                        >= RULE_EDGE):
                    first -= 1
                if (last < mask.shape[0] and closed[last, start:end].mean()
                        >= RULE_EDGE):
                    last += 1
            # Touching ink: one row beyond the line on either side, skipping
            # a row of blur
            touches = np.zeros(end - start, dtype=bool)
            for row in (first - 2, last + 1):
                if 0 <= row < mask.shape[0]:
                    touches |= mask[row, start:end]
            if touches.mean() <= RULE_CONTACT:
                mask[max(0, first - 1):last + 1, start:end] = False
    return ink

def stretches(flags):
    # The (start, end) of each run of True in a 1D boolean array
    padded = np.concatenate(([False], flags, [False])).astype(np.int8)
    change = np.diff(padded)
    return list(zip(np.nonzero(change == 1)[0], np.nonzero(change == -1)[0]))

def find_columns(ink):
    '''
    The (left, right) of each column of ink, left to right: stretches of
    pixel columns with ink, with touching columns split at their thinnest
    point and narrow stretches joined to a close neighbour or dropped.
    Returns them with the unit.
    '''
    profile = ink.sum(axis=0)
    if not profile.any():
        return [], 1
    spans = stretches(profile > GUTTER_INK * profile.max())
    # Noise makes narrow stretches, so the unit is the median width of the
    # stretches at least UNIT_SHARE as wide as the widest
    widths = [end - start for start, end in spans]
    unit = float(np.median([width for width in widths
                            if width >= UNIT_SHARE * max(widths)]))

    joined = []
    for start, end in spans:
        if (joined and end - start < MIN_COLUMN_WIDTH * unit
                and start - joined[-1][1] < MIN_COLUMN_WIDTH * unit):
            joined[-1] = (joined[-1][0], end)
        elif (joined and joined[-1][1] - joined[-1][0]
              < MIN_COLUMN_WIDTH * unit
              and start - joined[-1][1] < MIN_COLUMN_WIDTH * unit):
            joined[-1] = (joined[-1][0], end)
        else:
            joined.append((start, end))
    spans = [span for span in joined
             if span[1] - span[0] >= MIN_COLUMN_WIDTH * unit]

    split = []
    for start, end in spans:
        split.extend(split_span(profile, start, end, unit))
    return split, unit

def split_span(profile, start, end, unit):
    # A stretch of ink, split at its thinnest point while it's too wide
    if end - start <= SPLIT_WIDTH * unit:
        return [(start, end)]
    # Look for the split away from the edges, where a column's own ink is
    margin = round(0.5 * unit)
    middle = profile[start + margin:end - margin]
    if not len(middle):
        return [(start, end)]
    cut = start + margin + int(np.argmin(middle))
    if not start < cut < end:
        return [(start, end)]
    return (split_span(profile, start, cut, unit)
            + split_span(profile, cut, end, unit))

def find_words_in_column(ink, left, right, unit):
    '''
    The boxes of the words in one column of ink, top to bottom: stretches of
    rows with ink, joined across gaps under WORD_GAP units, with fragments
    joined to their nearer neighbour.
    '''
    column = ink[:, left:right]
    profile = column.sum(axis=1)
    pieces = [[start, end] for start, end in stretches(profile > 0)]

    def gap(index):
        # The gap below pieces[index]
        return pieces[index + 1][0] - pieces[index][1]

    # Join across small gaps
    index = 0
    while index < len(pieces) - 1:
        if gap(index) < WORD_GAP * unit:
            pieces[index][1] = pieces.pop(index + 1)[1]
        else:
            index += 1

    # Join fragments to the nearer neighbour
    def is_fragment(piece):
        start, end = piece
        return (end - start < FRAGMENT * unit
                or profile[start:end].sum() < MIN_INK * unit * unit)
    changed = True
    while changed and len(pieces) > 1:
        changed = False
        for index, piece in enumerate(pieces):
            if not is_fragment(piece):
                continue
            above = gap(index - 1) if index > 0 else math.inf
            below = gap(index) if index < len(pieces) - 1 else math.inf
            if min(above, below) > 2 * WORD_GAP * unit:
                continue
            if above <= below:
                pieces[index - 1][1] = pieces.pop(index)[1]
            else:
                start = pieces.pop(index)[0]
                pieces[index][0] = start
            changed = True
            break

    boxes = []
    for start, end in pieces:
        if profile[start:end].sum() < MIN_INK * unit * unit:
            continue
        ink_columns = np.nonzero(column[start:end].any(axis=0))[0]
        boxes.append([left + int(ink_columns[0]), int(start),
                      left + int(ink_columns[-1]) + 1, int(end)])
    return boxes

def find_words(image):
    '''
    Finds the words on a page, given as a PIL image, and returns a Page.
    '''
    grey = image.convert("L")
    angle = find_skew(binarize(grey))
    if abs(angle) >= FINE_STEP / 2:
        # The corners rotation adds are filled with paper, as white would
        # make the paper beside them look dark
        paper = int(np.median(np.asarray(grey)))
        grey = grey.rotate(angle, resample=Image.BICUBIC, expand=True,
                           fillcolor=paper)
    else:
        angle = 0.0
    ink = remove_rules(binarize(grey))
    spans, unit = find_columns(ink)
    words = []
    for column, (left, right) in enumerate(spans):
        boxes = find_words_in_column(ink, left, right, unit)
        words.extend({"box": box, "column": column, "position": position}
                     for position, box in enumerate(boxes))
    # Columns without words, of noise, don't count
    columns = sorted({word["column"] for word in words})
    renumber = {old: new for new, old in enumerate(columns)}
    for word in words:
        word["column"] = renumber[word["column"]]
    return Page(grey, ink, angle, unit, words, image.size)
