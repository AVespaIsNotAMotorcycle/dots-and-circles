import random
import sqlite3
import time
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np

import constants

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

def create_horizontal_image(word, font):
    image = Image.new(mode = "RGB", size = (HEIGHT, HEIGHT), color = (255,255,255))
    draw = ImageDraw.Draw(image)
    draw.text((10, HEIGHT / 2), word, fill=(0,0,0), font=font, anchor="lm")
    return image

def rotate_image(horizontal_image):
    left, top, right, bottom = get_boundaries()
    vertical_image = horizontal_image.rotate(-90).crop([left, top, right, bottom])
    return vertical_image

def create_lexigraph(word, font_index = get_random_font_index()):
    font = get_font(font_index)
    horizontal_image = create_horizontal_image(word, font)
    vertical_image = rotate_image(horizontal_image)
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
    for channel in pixel_tuple: total += int(channel)
    total = total / 3
    if total > 100: return 0
    return 1

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

def get_slices(font, manchu):
    lexigraph = create_lexigraph(manchu, font)
    array = image_to_array(lexigraph)
    slice_dimensions = get_slice_dimensions(font, manchu)
    row_labels = slice_dimensions_to_rows(manchu, slice_dimensions)

    slices = []
    non_blank_row_labels = []
    for i in range(350 - 30):
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
    # create_lexigraphy()
    get_slices(0, "ᠰᡳᠮᠨᡝᠪᡠᠮᠪᡳ")
