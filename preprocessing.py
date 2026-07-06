from PIL import Image, ImageDraw, ImageFont
import json
import random
import statistics

from utils import pixel_string_to_array

WIDTH = 50
HEIGHT = 350

COMPRESSED_WIDTH = 8
JOINT_COMPRESSED_WIDTH = COMPRESSED_WIDTH * 2 + 1

def colors(code):
    if code == 1: return (0, 0, 0)
    if code == 2: return (255, 0, 0)
    if code == 3: return (0, 255, 0)
    if code == 4: return (0, 0, 255)
    return (255, 255, 255)

def load_image(line_index):
    file = open("training_data.json", "r")
    lines = file.readlines()
    line = lines[line_index]

    line_json = json.loads(line)

    image_string = line_json["image"]
    return pixel_string_to_array(image_string)

'''
def center_line_boundaries(image_array):
    L_BUFFER = 1
    R_BUFFER = 0
    starts = []
    ends = []

    V_BUFFER = 10
    v_start = -1
    v_end = -1

    for row_index in range(HEIGHT):
        start = -1
        end = -1
        for col_index in range(WIDTH):
            y = row_index * WIDTH
            x = col_index
            pixel = image_array[x + y]

            if pixel == 0: continue

            if start == -1:
                start = x
                end = x
            else:
                end = x

        starts.append(start)
        ends.append(end)

        if v_start == -1 and start != -1:
            v_start = row_index
            v_end = row_index
        if v_start != -1 and start != -1:
            v_end = row_index

    starts = list(filter(lambda x: x >= 0, starts))
    ends = list(filter(lambda x: x >= 0, ends))
    
    if len(starts) == 0: return image_array
    if len(ends) == 0: return image_array

    median_start = int(statistics.median(starts))
    median_end = int(statistics.median(ends))
    return median_start, median_end
'''

def split_into_rows(image_array):
    image = []
    for row_index in range(HEIGHT):
        row_start = row_index * WIDTH
        row_end = row_start + WIDTH
        row = image_array[row_start:row_end]
        image.append(row)
    return image

def average_dark_pixels(image_array):
    rows = split_into_rows(image_array)

    total_dark_pixels = 0
    total_dark_rows = 0

    for row in rows:
        is_dark = False
        for pixel in row:
            if pixel != 0:
                is_dark = True
                total_dark_pixels += 1
        if is_dark:
            total_dark_rows += 1

    return total_dark_pixels / total_dark_rows

def horizontal_word_boundaries(image_array):
    rows = split_into_rows(image_array)
    start = 350
    end = 0
    for index, row in enumerate(rows):
        if 1 in row:
            if index < start: start = index
            if index > end: end = index
    return start, end

def get_buffers(image_array):
    start_row, end_row = horizontal_word_boundaries(image_array)
    length = end_row - start_row

    top_buffer = min(length * 0.1, 10)
    bottom_buffer = min(length * 0.25, 25)

    return top_buffer, bottom_buffer

def center_line_boundaries(image_array):
    start_row, end_row = horizontal_word_boundaries(image_array)
    top_buffer, bottom_buffer = get_buffers(image_array)

    columns = {}
    for row_index in range(HEIGHT):
        if row_index < start_row + top_buffer: continue
        if row_index > end_row - bottom_buffer: continue
        row_start = row_index * WIDTH
        row_end = row_start + WIDTH
        row = image_array[row_start:row_end]
        for index, pixel in enumerate(row):
            if index not in columns.keys():
                columns[index] = 0
            if pixel != 0: columns[index] += 1
    ranked = sorted(columns, key=columns.get, reverse=True)

    highest_count = columns[ranked[0]]
    arbitrary_range = 3
    cutoff = max(int(highest_count * 0.35), arbitrary_range)

    darkest_columns = []

    for column in ranked:
        if columns[column] < highest_count - cutoff: break
        if column + 1 not in darkest_columns and column - 1 not in columns: break
        darkest_columns.append(column)
    darkest_columns = sorted(darkest_columns)
    return darkest_columns[0], darkest_columns[len(darkest_columns) - 1]

def mark_center_line(image_array, color_code = 2):
    median_start, median_end = center_line_boundaries(image_array)

    marked = image_array.copy()
    for row_index in range(HEIGHT):
        for col_index in range(WIDTH):
            y = row_index * WIDTH
            x = col_index

            if row_index < v_start + V_BUFFER: continue
            if row_index > v_end - V_BUFFER: continue
            if col_index < median_start + L_BUFFER: continue
            if col_index > median_end - R_BUFFER: continue
            marked[x + y] = color_code
    return marked

def remove_row_center_line(row, start, end, highlight=False):
    extends_beyod = row[start - 1] != 0 or row[end + 1] != 0
    if extends_beyod: return row

    center = row[start:end+1]
    invert = []
    for pixel in center:
        if not highlight:
            invert.append(0)
            continue
        if pixel == 1: invert.append(2)
        else: invert.append(3)
    centerless_row = row[0:start] + invert + row[end+1:]
    return centerless_row

def remove_center_line(image_array, highlight=False):
    centerless_image = []
    center_line_start, center_line_end = center_line_boundaries(image_array)

    start_row, end_row = horizontal_word_boundaries(image_array)

    for row_index in range(HEIGHT):
        row_start = row_index * WIDTH
        row_end = row_start + WIDTH
        row = image_array[row_start:row_end]
        if row_index >= end_row - 10:
            centerless_image += row
        else:
            centerless_row = remove_row_center_line(row, center_line_start, center_line_end, highlight)
            centerless_image += centerless_row
    return centerless_image

def render(image_array, width = WIDTH):
    image = Image.new(mode = "RGB", size = (width, HEIGHT), color = (255, 255, 255))
    draw = ImageDraw.Draw(image)

    for index in range(len(image_array)):
        pixel = image_array[index]
        if pixel == 0: continue

        x = index % width
        y = int((index - x) / width)
        draw.point([x,y], fill = colors(pixel))
    image.show()

def preprocess(image_array, show = False):
    if show: render(image_array)
    if show: render(mark_center_line(image_array))
    if show: render(remove_center_line(image_array))
    return remove_center_line(image_array)

def start_of_color_block(image_array, color_code):
    start = -1
    end = -1
    
    for row_index in range(HEIGHT):
        for col_index in range(WIDTH):
            y = row_index * WIDTH
            x = col_index
            pixel = image_array[x + y]

            if pixel != color_code:
                if end > start: return [start, end]
                else: continue

            if start == -1: start = x
            if x > end: end = x
    return [start, end]

def extrema_black_columns(image_array, width):
    leftmost = width
    rightmost = 0
    for row_index in range(HEIGHT):
        for col_index in range(width):
            y = row_index * width
            x = col_index
            pixel = image_array[x + y]

            if pixel == 1:
                if x < leftmost:
                    leftmost = x
                if x > rightmost:
                    rightmost = x
    return [leftmost, rightmost]

def compress_row(row):
    if len(row) == COMPRESSED_WIDTH:
        return row
    if 1 not in row:
        return [0] * COMPRESSED_WIDTH

    compressed_row = []

    if len(row) > COMPRESSED_WIDTH:
        scale = len(row) / COMPRESSED_WIDTH
        for i in range(COMPRESSED_WIDTH):
            start = int(i * scale)
            end = int((i + 1) * scale)
            chunk = row[start:end]

            if 1 in chunk: compressed_row.append(1)
            else: compressed_row.append(0)
    else:
        for pixel in row:
            compressed_row.append(pixel)
        while len(compressed_row) < COMPRESSED_WIDTH:
            compressed_row.append(0)

    return compressed_row

def compress_half(image_array, width):
    extrema = extrema_black_columns(image_array, width)
    compressed = []
    left = extrema[0]
    right = extrema[1] + 1

    for row_index in range(HEIGHT):
        row = []
        for col_index in range(width):
            y = row_index * width
            x = col_index
            pixel = image_array[x + y]
            row.append(pixel)
        compressed_row = compress_row(row[left:right])
        for pixel in compressed_row:
            compressed.append(pixel)

    return compressed

def join_compressed_halves(left, right):
    joint = []
    for y in range(HEIGHT):
        start = y * COMPRESSED_WIDTH
        end = start + COMPRESSED_WIDTH
        for pixel in left[start:end]: joint.append(pixel)
        joint.append(0)
        for pixel in right[start:end]: joint.append(pixel)
    return joint

def split_halves(image_array, boundaries):
    marked = image_array.copy()

    left_half = []
    left_width = boundaries[1] - boundaries[0]

    right_half = []
    right_width = boundaries[3] - boundaries[2] - 1

    for row_index in range(HEIGHT):
        for col_index in range(WIDTH):
            y = row_index * WIDTH
            x = col_index
            pixel = image_array[x + y]

            if x >= boundaries[0] and x < boundaries[1]:
                left_half.append(pixel)
                if pixel == 0:
                    marked[x + y] = (3)
                    continue
            if x > boundaries[2] and x < boundaries[3]:
                right_half.append(pixel)
                if pixel == 0:
                    marked[x + y] = (4)
                    continue

    compressed_left = compress_half(left_half, left_width)
    compressed_right = compress_half(right_half, right_width)
    joint = join_compressed_halves(compressed_left, compressed_right)
    return joint

def compress(image_array, remove_center_line = True):
    marked_center = mark_center_line(image_array)
    center_line_boundaries = start_of_color_block(marked_center, 2)

    boundaries = [0,
                  center_line_boundaries[0],
                  center_line_boundaries[1],
                  WIDTH]
    compressed = split_halves(image_array, boundaries)
    return compressed

def identify_mark(image_array, starting_pixel):
    pixels_in_mark = []
    pixels_to_ignore = []
    frontier = [starting_pixel]

    while len(frontier) > 0:
        pixel = frontier.pop(0)
        if pixel in pixels_to_ignore:
            continue
        if pixel < 0:
            pixels_to_ignore.append(pixel)
            continue
        if pixel >= WIDTH * HEIGHT:
            pixels_to_ignore.append(pixel)
            continue
        if pixel in pixels_in_mark:
            pixels_to_ignore.append(pixel)
            continue
        if image_array[pixel] == 0:
            pixels_to_ignore.append(pixel)
            continue
        frontier.append(pixel)
        pixels_in_mark.append(pixel)
        neighbors = [pixel - WIDTH - 1, # top-left
                     pixel - WIDTH,     # top-center
                     pixel - WIDTH + 1, # top-right
                     pixel - 1,         # left
                     pixel + 1,         # right
                     pixel + WIDTH - 1, # bottom-left
                     pixel + WIDTH,     # bottom-center
                     pixel + WIDTH + 1] # bottom-right
        for neighbor in neighbors: frontier.append(neighbor)

    return pixels_in_mark

def render_mark(mark):
    image_array = [0] * WIDTH * HEIGHT
    for pixel in mark:
        image_array[pixel] = 1
    render(image_array)

def split_marks(image_array):
    marks = []
    pixels_already_claimed = []

    for index, color in enumerate(image_array):
        if color == 0: continue
        if index in pixels_already_claimed: continue
        pixels_in_mark = identify_mark(image_array, index)
        marks.append(pixels_in_mark)
        for pixel in pixels_in_mark: pixels_already_claimed.append(pixel)

    for mark in marks:
        render_mark(mark)

def identify_marks(image):
    # render(image)
    no_line = remove_center_line(image)
    render(no_line)
    marks = split_marks(no_line)

if __name__=="__main__":
    NUM_ENTRIES = 51358
    for i in range(1):
        index = random.randrange(NUM_ENTRIES)
        print(index)
        image_array = load_image(index)
        render(remove_center_line(image_array, True))
        identify_marks(image_array)
