from PIL import Image, ImageDraw, ImageFont

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

def create_lexigraph(word, font_index = 0):
    font = get_font(font_index)
    horizontal_image = create_horizontal_image(word, font)
    vertical_image = rotate_image(horizontal_image)
    return vertical_image

if __name__ == "__main__":
    lexigraph = create_lexigraph("ᠰᡳᠮᠨᡝᠪᡠᠮᠪᡳ")
    lexigraph.show()
