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
   columns touch. Then pieces of vertical lines broken into lengths too
   short for step 3, but long and thin for the page's unit, are removed
   too, along with what's beyond one close to the ink's edge (a book's
   fold strip, with its title and page number), and the columns found
   again.
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
# and at least this share of its extent the other way
RULE_ACROSS = 0.3
RULE_CONTACT = 0.1
# Breaks in a line up to this share of the ink's extent are bridged, and up
# to RULE_EDGE_ROWS rows either side inked along at least RULE_EDGE of the
# line are its blurred edges
RULE_GAP = 0.01
RULE_EDGE = 0.5
RULE_EDGE_ROWS = 2
# A frame's lines are within FRAME_BAND of the image's edges, and rows
# inside them inked along at least FRAME_HALO of them, up to
# FRAME_HALO_ROWS, are their ragged edges
FRAME_BAND = 0.25
FRAME_WIDTH = 0.02
FRAME_HALO = 0.15
FRAME_HALO_ROWS = 4
# Ink within this share of the image's shorter side of its edges is
# cleared, as the page's edge rather than text
BORDER = 0.01
# Binarizing: the radius of the blur against pixel noise, and ink pixels
# with fewer inked neighbours than SPECK_NEIGHBOURS are noise
NOISE_BLUR = 1
SPECK_NEIGHBOURS = 2
# Columns: the unit is measured on stretches of ink at least this share as
# wide as the widest
UNIT_SHARE = 0.3
# Columns: pixel columns with at least CORE_INK of the most ink are in a
# column's core; those with GUTTER_INK or less are blank, for measuring
# the unit
CORE_INK = 0.05
GUTTER_INK = 0.01
# A column reaches at most this many cores' widths beyond its core
COLUMN_REACH = 0.6
# A core whose most ink is under WEAK_CORE of the page's most is weak, and
# joins a neighbouring core less than JOIN_GAP cores' widths away
WEAK_CORE = 0.18
JOIN_GAP = 0.3
# A stretch of ink wider than this many units may be two columns touching,
# and is split if its ink, per pixel column, falls to SPLIT_VALLEY of the
# lower of the peaks either side
SPLIT_WIDTH = 1.6
SPLIT_VALLEY = 0.25
# Stretches narrower than this many units are noise or stray marks, unless
# they're close to a column
MIN_COLUMN_WIDTH = 0.25
# Words: gaps of at least WORD_GAP units separate words; pieces shorter
# than FRAGMENT units join their nearer neighbour, as do pieces with less
# than MIN_INK square units of ink
WORD_GAP = 0.08
# Boxes under SLIVER units tall and more than SLIVER_ASPECT times as wide
# are dropped
SLIVER = 0.15
SLIVER_ASPECT = 3
# At either end of a column, boxes under END_FLAT units tall and at least
# END_FLAT_ASPECT times as wide are dropped too
END_FLAT = 0.35
END_FLAT_ASPECT = 2.4
# Pieces shorter than this many units, with no ink on the stem, are
# fragments too
STEMLESS = 0.6
# Candidate breaks between tightly set words: the stem is followed in
# stretches of STEM_WINDOW units, a break is at least STEM_BREAK_ROWS rows
# without ink on it, and each part at least MIN_PART units tall
STEM_WINDOW = 1.0
STEM_BREAK_ROWS = 2
MIN_PART = 0.3
# Measuring strokes, ink is widened by this many units either side
STROKE_SWAY = 0.06
# Rows of a column with at most TRACE_INK units of ink, none of it within
# STEM_BAND units of the stem, are gap
TRACE_INK = 0.0
STEM_BAND = 0.08
FRAGMENT = 0.2
MIN_INK = 0.03
# Pieces of vertical lines: at least RULE_PIECE units long and at most
# RULE_PIECE_WIDTH units wide, gaps of up to RULE_PIECE_GAP pixels along
# them bridged, touched along at most RULE_CONTACT of their length. One
# within FOLD_STRIP units of the ink's edge bounds the text.
RULE_PIECE = 3.0
RULE_PIECE_WIDTH = 0.25
RULE_PIECE_GAP = 2
FOLD_STRIP = 1.5


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

    def straighten(self, image):
        '''
        The image as given (in colour, say) turned as the page was, so it
        lines up with self.image and the words' boxes. The corners the turn
        adds are filled with the image's paper colour.
        '''
        if not self.angle:
            return image.copy()
        pixels = np.asarray(image)
        paper = tuple(int(value) for value in np.median(
            pixels.reshape(-1, pixels.shape[-1]) if pixels.ndim == 3
            else pixels.reshape(-1, 1), axis=0))
        return image.rotate(self.angle, resample=Image.BICUBIC, expand=True,
                            fillcolor=paper if len(paper) > 1 else paper[0])

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
    # A slight blur first, so pixel noise on the paper doesn't cross the
    # threshold
    pixels = np.asarray(image.filter(ImageFilter.GaussianBlur(NOISE_BLUR)),
                        dtype=np.float64)
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
    RULE_LENGTH of the ink's extent along them and RULE_ACROSS of it across, and for vertical ones, with other ink
    touching their sides along at most RULE_CONTACT of their length. Small breaks in a line, from
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
        across = (columns[-1] - columns[0] if transposed
                  else rows[-1] - rows[0])
        closed = close_gaps(mask, max(2, round(RULE_GAP * extent)))
        line_rows, starts, ends = runs(closed)
        # Long against the page both ways: a narrow image of one word is
        # no wider than its teeth
        long = ((ends - starts >= RULE_LENGTH * extent)
                & (ends - starts >= RULE_ACROSS * across))
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
            # a row of blur. Only vertical lines need the test, as stems are
            # as long and straight; no text has a horizontal stroke this
            # long.
            touches = np.zeros(end - start, dtype=bool)
            for row in (first - 2, last + 1):
                if 0 <= row < mask.shape[0]:
                    touches |= mask[row, start:end]
            if not transposed or touches.mean() <= RULE_CONTACT:
                mask[max(0, first - 1):last + 1, start:end] = False
    return ink

def frame_lines(ink):
    '''
    Where a frame around the text is: (top, bottom, left, right), the first
    row or column inside each of its lines, or the image's edge where there
    is none. A frame line is a run of ink at least RULE_LENGTH of the ink's
    extent, within FRAME_BAND of the image's edge: the outermost there,
    with any others within FRAME_WIDTH of it (a frame drawn thick, or
    double), but not a ruled line further in, between two columns. Rows or columns just inside it inked
    along at least FRAME_HALO of it (its ragged or blurred edge) count as
    the line too, up to FRAME_HALO_ROWS of them.
    '''
    height, width = ink.shape
    rows = np.nonzero(ink.any(axis=1))[0]
    columns = np.nonzero(ink.any(axis=0))[0]
    bounds = [0, height, 0, width]
    if not len(rows):
        return bounds
    for transposed in (False, True):
        mask = ink.T if transposed else ink
        extent = (rows[-1] - rows[0] if transposed
                  else columns[-1] - columns[0])
        across = (columns[-1] - columns[0] if transposed
                  else rows[-1] - rows[0])
        closed = close_gaps(mask, max(2, round(RULE_GAP * extent)))
        line_rows, starts, ends = runs(closed)
        long = ((ends - starts >= RULE_LENGTH * extent)
                & (ends - starts >= RULE_ACROSS * across))
        size = mask.shape[0]
        near = line_rows[long & (line_rows < FRAME_BAND * size)]
        far = line_rows[long & (line_rows >= (1 - FRAME_BAND) * size)]
        # The line's span, to measure its halo along
        span = (slice(None) if not long.any()
                else slice(int(starts[long].min()), int(ends[long].max())))
        first, last = (2, 3) if transposed else (0, 1)
        if len(near):
            edge = int(near[near <= near.min() + FRAME_WIDTH * size].max()) + 1
            for _ in range(FRAME_HALO_ROWS):
                if edge < size and closed[edge, span].mean() >= FRAME_HALO:
                    edge += 1
            bounds[first] = edge
        if len(far):
            edge = int(far[far >= far.max() - FRAME_WIDTH * size].min())
            for _ in range(FRAME_HALO_ROWS):
                if edge > 0 and closed[edge - 1, span].mean() >= FRAME_HALO:
                    edge -= 1
            bounds[last] = edge
    return bounds

def rule_pieces(ink, unit):
    '''
    Pieces of vertical lines in the ink: [left, top, right, bottom] of each
    run of ink down the page at least RULE_PIECE units long, taken
    together with overlapping runs beside it, that's at most
    RULE_PIECE_WIDTH units wide and touched by other ink along at most
    RULE_CONTACT of its length. A frame's inner line is often broken into
    pieces too short for remove_rules. A Manchu stem is as thin, but
    shorter, or touched by its teeth.
    '''
    # Runs along mask's rows: down the page, one row per pixel column
    xs, starts, ends = runs(close_gaps(ink.T, RULE_PIECE_GAP))
    long = ends - starts >= RULE_PIECE * unit
    xs, starts, ends = xs[long], starts[long], ends[long]
    # Runs in neighbouring pixel columns that overlap are one piece
    parent = list(range(len(xs)))
    def root(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    at = {}
    for index, x in enumerate(xs):
        at.setdefault(int(x), []).append(index)
    for x, here in at.items():
        for index in here:
            for other in at.get(x - 1, []):
                if starts[index] < ends[other] and starts[other] < ends[index]:
                    parent[root(index)] = root(other)
    groups = {}
    for index in range(len(xs)):
        groups.setdefault(root(index), []).append(index)
    pieces = []
    for members in groups.values():
        left, right = int(xs[members].min()), int(xs[members].max()) + 1
        top, bottom = int(starts[members].min()), int(ends[members].max())
        if right - left > max(2, RULE_PIECE_WIDTH * unit):
            continue
        # Touching ink: one pixel column beyond either side, skipping a
        # column of blur, as remove_rules tests
        touches = np.zeros(bottom - top, dtype=bool)
        for x in (left - 2, right + 1):
            if 0 <= x < ink.shape[1]:
                touches |= ink[top:bottom, x]
        if touches.mean() <= RULE_CONTACT:
            pieces.append([left, top, right, bottom])
    return pieces

def remove_rule_pieces(ink, unit):
    '''
    Removes rule_pieces from the ink, in place, with a pixel column of blur
    either side. One within FOLD_STRIP units of the ink's left or right
    edge bounds the text, so the ink beyond it goes too. Returns whether
    there were any.
    '''
    pieces = rule_pieces(ink, unit)
    if not pieces:
        return False
    columns = np.nonzero(ink.any(axis=0))[0]
    first, last = int(columns[0]), int(columns[-1]) + 1
    for left, top, right, bottom in pieces:
        ink[top:bottom, max(0, left - 1):right + 1] = False
        if last - right <= FOLD_STRIP * unit:
            ink[:, max(0, left - 1):] = False
        elif left - first <= FOLD_STRIP * unit:
            ink[:, :right + 1] = False
    return True

def stretches(flags):
    # The (start, end) of each run of True in a 1D boolean array
    padded = np.concatenate(([False], flags, [False])).astype(np.int8)
    change = np.diff(padded)
    return list(zip(np.nonzero(change == 1)[0], np.nonzero(change == -1)[0]))

def find_columns(ink):
    '''
    The (left, right) of each column of ink, left to right, the unit, and
    the (left, right) of narrow stray marks, such as what's left of ruled
    lines.

    Gutters aren't always blank: paper specks, and small glosses set
    between the columns, leave some ink in them. So a column's core is a
    stretch of pixel columns with at least CORE_INK of the most ink, and
    the gutter between two cores is shared at its least inked point, so no
    column loses the tips of its teeth. Narrow cores are joined to a close
    neighbour or dropped, and cores too wide for one column are split
    where their ink dips.
    '''
    profile = ink.sum(axis=0)
    if not profile.any():
        return [], 1, []
    smooth = np.convolve(profile, np.ones(3) / 3, mode="same")
    cores = stretches(smooth > CORE_INK * smooth.max())
    widths = [end - start for start, end in cores]
    core_unit = float(np.median([width for width in widths
                                 if width >= UNIT_SHARE * max(widths)]))

    # A weak core (marks standing out from a stem, in some typefaces) right
    # beside a strong one is part of its column. Small glosses between
    # columns are as weak but stand clear of them.
    def weak(core):
        return smooth[core[0]:core[1]].max() < WEAK_CORE * smooth.max()
    joined = []
    for core in cores:
        if joined and core[0] - joined[-1][1] < JOIN_GAP * core_unit \
                and (weak(core) or weak(joined[-1])):
            joined[-1] = (joined[-1][0], core[1])
        else:
            joined.append(core)
    # Narrow cores standing alone aren't columns: specks, or what's left of
    # a ruled line or the page's edge. find_words erases them.
    lines = [core for core in joined
             if core[1] - core[0] < MIN_COLUMN_WIDTH * core_unit]
    cores = [core for core in joined if core not in lines]
    if not cores:
        return [], 1, lines
    split = []
    for start, end in cores:
        split.extend(split_span(profile, start, end, core_unit))
    cores = split

    # Share each gutter at its least inked point
    inked = np.nonzero(profile)[0]
    bounds = [int(inked[0])]
    for (_, left_end), (right_start, _) in zip(cores, cores[1:]):
        bounds.append(left_end + int(np.argmin(smooth[left_end:right_start]))
                      if right_start > left_end else left_end)
    bounds.append(int(inked[-1]) + 1)
    # But no further than COLUMN_REACH cores' widths beyond its core, so
    # stray ink in a wide margin (what's left of a frame, say) is in no
    # column
    reach = COLUMN_REACH * core_unit
    spans = [(max(start, int(core[0] - reach)),
              min(end, int(core[1] + reach)))
             for (start, end), core in zip(zip(bounds, bounds[1:]), cores)]

    # The unit is measured on each column's ink, as on a page with blank
    # gutters: from its first to last pixel column with more than a trace
    extents = []
    for start, end in spans:
        inked = np.nonzero(smooth[start:end] > GUTTER_INK * smooth.max())[0]
        if len(inked):
            extents.append(inked[-1] - inked[0] + 1)
    unit = float(np.median([width for width in extents
                            if width >= UNIT_SHARE * max(extents)]))
    return spans, unit, lines

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
    # Touching columns meet in a valley well below both their peaks; one
    # wide column (such as a Manchu one among narrower Chinese ones) has
    # no such valley
    if profile[cut] > SPLIT_VALLEY * min(profile[start:cut].max(),
                                         profile[cut:end].max()):
        return [(start, end)]
    return (split_span(profile, start, cut, unit)
            + split_span(profile, cut, end, unit))

def find_words_in_column(ink, left, right, unit):
    '''
    The boxes of the words in one column of ink, top to bottom: stretches of
    rows with more than a trace of ink, joined across gaps under WORD_GAP
    units, with fragments joined to their nearer neighbour.

    In print, words are often closer than a blank row: punctuation, or a
    stray mark, crosses the gap. So a row also counts as gap when it has at
    most TRACE_INK units of ink and none of it on the stem (within
    STEM_BAND units of the pixel column with the most ink): inside a word
    the stem is continuous, though in some typefaces the strokes along it
    are as thin as a trace. Each word then reaches into its gaps as far as
    their least inked rows, so the thin tips of its letters stay in its
    box.
    '''
    column = ink[:, left:right]
    profile = column.sum(axis=1)
    stem = int(np.argmax(column.sum(axis=0)))
    band = max(1, round(STEM_BAND * unit))
    on_stem = column[:, max(0, stem - band):stem + band + 1].any(axis=1)
    gap_rows = (profile == 0) | ((profile <= TRACE_INK * unit) & ~on_stem)
    pieces = [[start, end] for start, end in stretches(~gap_rows)]

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

    # Join fragments to the nearer neighbour: pieces too short or faint to
    # be a word, and short pieces off the stem (a tail curving away from
    # it, or punctuation beside it), since every word has a stem
    def is_fragment(piece):
        start, end = piece
        return (end - start < FRAGMENT * unit
                or profile[start:end].sum() < MIN_INK * unit * unit
                or (end - start < STEMLESS * unit
                    and not on_stem[start:end].any()))
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

    # Reach into the gaps, up to their least inked rows, and as far as the
    # ink goes at the top and bottom
    inked = np.nonzero(profile)[0]
    if not len(inked):
        return []
    edges = [int(inked[0])]
    for (_, above_end), (below_start, _) in zip(pieces, pieces[1:]):
        gap_profile = profile[above_end:below_start]
        lowest = np.nonzero(gap_profile == gap_profile.min())[0]
        # The middle of the least inked rows
        edges.append(above_end + int(lowest[len(lowest) // 2]))
    edges.append(int(inked[-1]) + 1)
    pieces = [[edges[index], edges[index + 1]]
              for index in range(len(pieces))]

    boxes = []
    for start, end in pieces:
        if profile[start:end].sum() < MIN_INK * unit * unit:
            continue
        rows = np.nonzero(column[start:end].any(axis=1))[0]
        start, end = start + int(rows[0]), start + int(rows[-1]) + 1
        # Flat slivers (what's left of a broken frame line) aren't text:
        # Manchu words are tall, and Chinese characters about square
        ink_columns = np.nonzero(column[start:end].any(axis=0))[0]
        if (end - start < SLIVER * unit and ink_columns[-1] - ink_columns[0]
                + 1 > SLIVER_ASPECT * (end - start)):
            continue
        boxes.append([left + int(ink_columns[0]), int(start),
                      left + int(ink_columns[-1]) + 1, int(end)])
    # A flat piece at either end of a column is what's left of a frame
    # line, which a column of text runs up to
    def flat(box):
        width, height = box[2] - box[0], box[3] - box[1]
        return height < END_FLAT * unit and width >= END_FLAT_ASPECT * height
    while boxes and flat(boxes[-1]):
        boxes.pop()
    while boxes and flat(boxes[0]):
        boxes.pop(0)
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
    ink = binarize(grey)
    # Only the text inside a frame, if there is one
    top, bottom, left, right = frame_lines(ink)
    ink[:top] = ink[bottom:] = False
    ink[:, :left] = ink[:, right:] = False
    ink = remove_rules(ink)
    # The edges of a photo or scan often hold the page's edge or shadow,
    # not text
    border = max(2, round(BORDER * min(ink.shape)))
    ink[:border] = ink[-border:] = False
    ink[:, :border] = ink[:, -border:] = False
    spans, unit, lines = find_columns(ink)
    if remove_rule_pieces(ink, unit):
        # The unit stays as first measured: without a fold strip's narrow
        # column, it can come out different, and with it the words
        spans, _, lines = find_columns(ink)
    for left, right in lines:
        ink[:, left:right] = False
    words = []
    for column, (left, right) in enumerate(spans):
        boxes = find_words_in_column(ink, left, right, unit)
        # Plain ints, so boxes can go straight into JSON
        words.extend({"box": [int(value) for value in box],
                      "column": column, "position": position}
                     for position, box in enumerate(boxes))
    # Columns without words, of noise, don't count
    columns = sorted({word["column"] for word in words})
    renumber = {old: new for new, old in enumerate(columns)}
    for word in words:
        word["column"] = renumber[word["column"]]
    return Page(grey, ink, angle, float(unit), words, image.size)

def longest_stroke(ink, box, unit):
    '''
    The length, in pixels, of the longest vertical run of ink in a box,
    bridging one-pixel breaks. The ink is first widened by STROKE_SWAY
    units either side, so a stem that sways, as in cursive typefaces,
    still counts as one run. Manchu stems run for several units; small
    Chinese glosses have no stroke that long.
    '''
    left, top, right, bottom = box
    region = ink[top:bottom, left:right]
    sway = max(1, round(STROKE_SWAY * unit))
    padded = np.pad(region, ((0, 0), (sway, sway)))
    widened = np.zeros_like(region)
    for shift in range(2 * sway + 1):
        widened |= padded[:, shift:shift + region.shape[1]]
    _, starts, ends = runs(close_gaps(widened.T, 1))
    return int((ends - starts).max()) if len(starts) else 0

def stem_breaks(ink, box, unit):
    '''
    Where a box of Manchu might hold two words: rows (counted from the
    box's top) in the middle of each stretch of at least STEM_BREAK_ROWS
    rows with no ink on the stem, at least MIN_PART units from either end.
    In tightly set print a word's tail runs alongside the next word's
    start, so no row is blank between them, but the stem breaks; it also
    breaks inside some letters, so these are only candidates (see
    ocr.page_reader.split_at_stem_breaks). The stem is followed down the
    box, as the most inked pixel column in each stretch of STEM_WINDOW
    units.
    '''
    left, top, right, bottom = box
    region = ink[top:bottom, left:right]
    height = bottom - top
    window = max(1, round(STEM_WINDOW * unit))
    band = max(1, round(STEM_BAND * unit))
    on_stem = np.zeros(height, dtype=bool)
    for start in range(0, height, window):
        rows = region[start:start + window]
        if not rows.any():
            continue
        stem = int(np.argmax(rows.sum(axis=0)))
        on_stem[start:start + window] = rows[
            :, max(0, stem - band):stem + band + 1].any(axis=1)
    return [int(start + end) // 2 for start, end in stretches(~on_stem)
            if end - start >= STEM_BREAK_ROWS
            and start >= MIN_PART * unit and height - end >= MIN_PART * unit]
