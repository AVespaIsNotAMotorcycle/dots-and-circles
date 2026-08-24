import random
import sqlite3
import time
import json
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageFilter
import numpy as np

import constants
import corpus

constants.ROW_LENGTH = 50
HEIGHT = 350

FONTS = ["fonts/XM_BiaoHei.ttf",
         "fonts/XM_GuFeng.ttf",
         "fonts/XM_LiuYe.ttf",
         "fonts/XM_ShuKai.ttf",
         "fonts/XM_WenJian.ttf",
         "fonts/XM_WenQin.ttf",
         "fonts/XM_XingShu.ttf",
         "fonts/XM_YaBai.ttf",
         "fonts/XM_YingBi.ttf",
         "fonts/XM_ZhengBai.ttf",
         "fonts/XM_ZhengHei.ttf"]

def get_fonts_dict():
    fonts_dict = {}
    for index, fontname in enumerate(FONTS):
        fonts_dict[str(index)] = fontname
    return fonts_dict

def get_font_filename(index):
    return FONTS[index]

def get_random_font_index():
    return random.randrange(len(FONTS))

def get_font(font_index = 0):
    font_name = FONTS[int(font_index)]
    font = ImageFont.truetype(font_name, 30)
    return font

def get_boundaries():
    left = (HEIGHT - constants.ROW_LENGTH) / 2
    top = 0
    right = left + constants.ROW_LENGTH
    bottom = HEIGHT
    return [left, top, right, bottom]

def create_horizontal_image(word, font, seed=0):
    image = Image.new(mode = "RGB", size = (HEIGHT, HEIGHT), color = (255,255,255)) \

    draw = ImageDraw.Draw(image)

    offset_x = 0 if seed == None else (seed % 12) - 6
    offset_y = 0 if seed == None else seed**2 % 10
    position = (10 + offset_y, HEIGHT / 2 + offset_x)

    draw.text(position, word, fill=(0,0,0), font=font, anchor="lm")
    return image

def rotate_image(horizontal_image):
    left, top, right, bottom = get_boundaries()
    vertical_image = horizontal_image.rotate(-90).crop([left, top, right, bottom])
    return vertical_image

def crop_image(vertical_image, seed=0):
    image_array = image_to_array(vertical_image)
    start_of_whitespace = len(image_array) - 1
    while sum(image_array[start_of_whitespace]) == 0:
        start_of_whitespace -= 1
    image_array = image_array[:start_of_whitespace + 10]
    image_array = image_array * 255
    cropped_image = Image.fromarray(image_array.astype('uint8'))
    cropped_image = ImageOps.invert(cropped_image)

    cropped_image = cropped_image.convert(mode="RGBA")
    width, height = cropped_image.size
    text_alpha = 255 - (seed**2 % 150) if seed != 0 else 255
    for y in range(height):
        for x in range(width):
            r, g, b, a = cropped_image.getpixel((x, y))
            new_alpha = int(255 - ((r + g + b) / 3)) \
                    if r + g + b > 0 \
                    else text_alpha
            cropped_image.putpixel((x, y), (r, g, b, new_alpha))

    background_id = seed % 16
    background = Image.new(mode = "RGB", size = (HEIGHT, HEIGHT), color = (255,255,255)) \
            if background_id == 0 \
            else Image.open(f"paper_textures/{background_id:02d}.jpg")
    if seed != 0:
        b_w, b_h = background.size
        divisor = 2 if background_id < 6 else 8
        background.resize((int(b_w / divisor), int(b_h / divisor)))
    b_w, b_h = background.size
    range_x, range_y = b_w - width, b_h - height
    offset_x, offset_y = seed ** 3 % range_x, seed ** 4 % range_y
    background = background.crop((0 + offset_x, 0 + offset_y, width + offset_x, height + offset_y))
    background = background.convert(mode="RGBA")

    image = Image.alpha_composite(background, cropped_image)
    return image

def create_lexigraph(word, font_index = get_random_font_index(), crop=False, seed=None):
    actual_seed = random.randint(1, 1000) if seed == None else seed
    font = get_font(font_index)
    horizontal_image = create_horizontal_image(word, font, seed=actual_seed)
    vertical_image = rotate_image(horizontal_image)
    if crop: return crop_image(vertical_image, seed=actual_seed)
    return vertical_image

def save_lexigraph(font, manchu, boundaries):
    timestamp = time.time()
    data = [str(manchu), int(font), json.dumps(boundaries), str(timestamp)]

    connection = sqlite3.connect(constants.DB_NAME)
    cursor = connection.cursor()

    success = True
    try: cursor.executemany("INSERT INTO lexigraphy VALUES (?, ?, ?, ?)", [data])
    except: success = False

    connection.commit()
    connection.close()
    return success

def get_slice_dimensions(font, manchu):
    connection = sqlite3.connect(constants.DB_NAME)
    cursor = connection.cursor()

    command = "SELECT boundaries FROM lexigraphy WHERE font={0} AND manchu=\"{1}\"".format(font, manchu)
    row = cursor.execute(command).fetchone()
    connection.close()

    if row == None: return []
    else: return json.loads(row[0])

def collapse_pixel(pixel_tuple):
    total = 0
    if type(pixel_tuple) == np.uint8: return pixel_tuple
    for channel in pixel_tuple: total += int(channel)
    total = total / 3
    return (255 - total) / 255

def image_to_array(image):
    image_array = np.array(image)
    array = np.zeros((350, 50))
    for y, row in enumerate(image_array):
        for x, pixel in enumerate(row):
            array[y][x] = collapse_pixel(pixel)
    return array

def slice_dimensions_to_rows(manchu, slice_dimensions):
    rows = []

    for index, slice in enumerate(slice_dimensions):
        letter = manchu[index]
        margin = int(slice[0])
        length = int(slice[1])
        for i in range(margin):
            if index == 0: rows.append(' ')
            else: rows.append('*')
        for i in range(length): rows.append(letter)
    while len(rows) < 330: rows.append(' ')

    return rows[10:]

def completely_blank(slice):
    for row in slice:
        for pixel in row:
            if pixel != 0:
                return False
    return True

def get_lexigraph_array(font, manchu):
    lexigraph = create_lexigraph(manchu, font)
    array = image_to_array(lexigraph)
    return array

def lighten_edges(image):
    w, h = image.size
    for x in range(w):
        for y in range(h):
            pixel = (x, y)
            depth_x = min(x, w - x)
            depth_y = min(y, h - y)
            delta = min(depth_x, depth_y) + 1

            val = image.getpixel(pixel)
            if val == 255: continue

            n_val = 255 - (delta * 24 - 1)
            if n_val < 0: continue
            
            n_val += val
            if n_val > 255: n_val = 255
            image.putpixel(pixel, n_val)
    
    return image

def preprocess_image(image):
    image = image.convert(mode="L")
    image = ImageOps.autocontrast(image, (0, 20))
    image = ImageOps.autocontrast(image, (0, 40))
    image = ImageOps.autocontrast(image, (5, 60))
    image = ImageOps.autocontrast(image, (5, 80))
    image = lighten_edges(image)
    return image

def get_slices(font, manchu):
    lexigraph = create_lexigraph(manchu, font, crop=True)
    lexigraph = preprocess_image(lexigraph)
    _, height = lexigraph.size
    array = image_to_array(lexigraph)
    slice_dimensions = get_slice_dimensions(font, manchu)
    row_labels = slice_dimensions_to_rows(manchu, slice_dimensions)

    slices = []
    non_blank_row_labels = []
    for i in range(height - 20):
        start = i + 10
        end = start + constants.NUMBER_OF_ROWS
        slice = array[start:end]
        if completely_blank(slice): continue
        slices.append(slice.reshape(constants.INPUT_LAYER_SIZE, 1))
        non_blank_row_labels.append(row_labels[i])

    return slices, non_blank_row_labels

def get_lexigraph_page(start, end):
    connection = sqlite3.connect(constants.DB_NAME)
    cursor = connection.cursor()

    all_characters = []
    rows = cursor.execute("SELECT manchu, font, boundaries FROM lexigraphy").fetchall()
    page = rows[start:end]

    connection.close()
    return page

def get_random_marked_lexigraph():
    connection = sqlite3.connect(constants.DB_NAME)
    cursor = connection.cursor()

    all_characters = []
    rows = cursor.execute("SELECT manchu, font FROM lexigraphy").fetchall()
    random.shuffle(rows)
    row = rows[0]
    manchu, font = row
    slices, row_labels = get_slices(font, manchu)

    connection.close()
    return font, manchu, slices, row_labels

def drop_table(cursor):
    try: cursor.execute("DROP TABLE lexigraphy")
    except: return

def create_table(cursor):
    columns = ",".join(["manchu TEXT NOT NULL",
                        "font INTEGER NOT NULL",
                        "boundaries TEXT NOT NULL",
                        "timestamp INTEGER NOT NULL",
                        "PRIMARY KEY(manchu, font)"])
    cursor.execute("CREATE TABLE lexigraphy({0})".format(columns))

def create_lexigraphy():
    connection = sqlite3.connect(constants.DB_NAME)
    cursor = connection.cursor()

    drop_table(cursor)
    create_table(cursor)

    connection.commit()
    connection.close()

if __name__ == "__main__":
    for i in range(1):
        # get_random_marked_lexigraph()
        lexigraph = get_slices(0, corpus.get_random_word()['manchu'])
