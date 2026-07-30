import random
import sqlite3
import time
import json
from PIL import Image, ImageDraw, ImageFont

from db import get_db_name

WIDTH = 50
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
    font_name = FONTS[font_index]
    font = ImageFont.truetype(font_name, 30)
    return font

def get_boundaries():
    left = (HEIGHT - WIDTH) / 2
    top = 0
    right = left + WIDTH
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

    connection = sqlite3.connect(get_db_name())
    cursor = connection.cursor()

    success = True
    try: cursor.executemany("INSERT INTO lexigraphy VALUES (?, ?, ?, ?)", [data])
    except: success = False

    connection.commit()
    connection.close()
    return success

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
    connection = sqlite3.connect(get_db_name())
    cursor = connection.cursor()

    drop_table(cursor)
    create_table(cursor)

    connection.commit()
    connection.close()

if __name__ == "__main__":
    create_lexigraphy()
