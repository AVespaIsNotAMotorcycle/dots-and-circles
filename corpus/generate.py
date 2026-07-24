from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import json
import unicodedata
import sys

width = 50
height = 350

FONTS = ["XM_BiaoHei.ttf",
         "XM_GuFeng.ttf",
         "XM_LiuYe.ttf",
         "XM_ShuKai.ttf",
         "XM_WenJian.ttf",
         "XM_WenQin.ttf",
         "XM_XingShu.ttf",
         "XM_YaBai.ttf",
         "XM_YingBi.ttf",
         "XM_ZhengBai.ttf",
         "XM_ZhengHei.ttf"]

def is_mongolian(char):
    block = unicodedata.name(char).split()[0]
    return block == 'MONGOLIAN'

def remove_odd_characters(input_string):
    ODD_CHARACTERS = [
        '\t',
        '\n',
        '/',
        '(',
        ')',
        '[',
        ']',
        '!',
        '︕',
        '?',
        '0',
        '1',
        '2',
        '3',
        '4',
        '5',
        '6',
        '7',
        '8',
        '9',
    ]
    new_string = input_string
    for character in ODD_CHARACTERS:
        new_string = new_string.replace(character, ' ')

    return new_string

def mongolian_only(string):
    new_string = ""
    for char in string:
        if is_mongolian(char):
            new_string += char
    return new_string

def split_string(input_string):
    new_string = remove_odd_characters(input_string)
    was_split = new_string != input_string
    split = new_string.split()
    output = []
    for item in split:
        output.append(mongolian_only(item))
    return output

def has_hanzi(word):
    hanzi_blocks = ["KATAKANA", "CJK"]
    for char in word:
        block = unicodedata.name(char).split()[0]
        for hanzi_block in hanzi_blocks:
            if block == hanzi_block:
                return True
    return False

def read_corpus():
    file = open("db.json", "r", encoding="utf-8")
    entries = file.readlines()
    file.close()
    return entries

def generate_random_image():
    entries = read_corpus()
    index = random.randrange(len(entries))
    entry = json.loads(entries[index])
    abkai = split_string(entry['r'])
    words = split_string(entry['m'])
    for word in words:
        generate_images(word, random.randrange(len(FONTS)))
    print(words, abkai)

def collapse_channels(pixel):
    channel_1 = pixel[0] / 255
    channel_2 = pixel[1] / 255
    channel_3 = pixel[2] / 255

    total_value = channel_1 + channel_2 + channel_3
    average = total_value / 3

    # 1 is white
    # 0 is black
    # anything between is grey
    # sensitivity is the cutoff for coercing a grey pixel to black
    sensitivity = 0.5
    if average > sensitivity: return int(0)
    else: return int(1)

def simplify_array(array):
    simplified = []

    for row_index in range(len(array)):
        row = array[row_index]
        new_row = []
        for cell_index in range(len(row)):
            cell = row[cell_index]
            collapsed = collapse_channels(cell)
            new_row.append(collapsed)
        simplified.append(new_row)
    return simplified

def get_unique_words(start_index, end_index):
    words = []
    entries = read_corpus()
    percent_per_entry = 1 / len(entries)
    for index in range(start_index, end_index):
        entry = json.loads(entries[index])
        manchu = split_string(entry['m'])
        for word in manchu:
            if word not in words: words.append(word)
    return words

def image_array_to_string(array):
    string = ""
    empty_space = ""
    for row in array:
        for integer in row:
            if integer > 0:
                string += empty_space
                empty_space = ""
                string += '1'
            else:
                empty_space += '0'
    return string

def generate_training_data(start_index = 0, end_index = -1):
    entries = read_corpus()

    print("Assembling list of unique words...")
    words = get_unique_words(0, len(entries))
    word_count = len(words)
    print("Found {0} unique words.".format(word_count))
    file = open("../training_data.json", "w", encoding="utf-8")

    if end_index == -1: end_index = word_count
    if end_index < start_index + 1: end_index = start_index + 1
    if end_index > word_count: end_index = word_count

    for index in range(start_index, end_index):
        word = words[index]
        images = generate_images(word)
        for image in images:
            image_array = np.array(image)
            simple = simplify_array(image_array)
            file.write(json.dumps({ "image": image_array_to_string(simple), "word": word }) + '\n')

        count_string = str(word_count)
        index_string = str(index + 1).rjust(len(count_string), " ")
        percent = int((index - start_index) / (end_index - start_index) * 100)
        percent_string = "{0}%".format(percent).rjust(6, " ")
        print("{0} / {1} | {2} | {3}".format(index_string, count_string, percent_string, word))

    file.close()

def generate_image(word, font = 0):
    font_name = FONTS[font]
    font = ImageFont.truetype(font_name, 30)

    image = Image.new(mode = "RGB", size = (height, height), color = (255,255,255))
    draw = ImageDraw.Draw(image)
    draw.text((10, height / 2), word, fill=(0,0,0), font=font, anchor="lm")

    left = (height - width) / 2
    top = 0
    right = left + width
    bottom = height
    image2 = image.rotate(-90).crop([left, top, right, bottom])
    # image2.show()
    return image2

def generate_images(word, font = -1):
    images = []

    if font >= 0 and font < len(FONTS):
        images.append(generate_image(word, font))
        return images

    for i in range(len(FONTS)):
        images.append(generate_image(word, i))
    return images

def find_longest_word():
    longest_indices = [0, 0]
    longest_length = 0
    longest_word = ""

    entries = read_corpus()
    for entry_index in range(len(entries)):
        entry = json.loads(entries[entry_index])
        manchu_words = split_string(entry['m'])

        for word_index in range(len(manchu_words)):
            word = manchu_words[word_index]

            if has_hanzi(word): continue
            
            length = len(word)
            if length > longest_length:
                longest_length = length
                longest_indices = [entry_index, word_index]
                longest_word = word
                
    print('\n')
    print(entries[longest_indices[0]])
    print('\n')
    print(longest_indices, longest_length)
    print('\n')
    print(longest_word)
    print('\n')
    print("Has Hanzi: ", has_hanzi(longest_word))

    return longest_indices

def find_all_unique_characters():
    characters = []

    entries = read_corpus()
    for entry_index in range(len(entries)):
        entry = json.loads(entries[entry_index])
        manchu_words = split_string(entry['m'])
        for word in manchu_words:
            for character in word:
                if (has_hanzi(character)): continue
                if character not in characters:
                    characters.append(character)

    print('\n', characters, '\n', len(characters), '\n')
    for char in characters:
        print(char, unicodedata.name(char))

generate_training_data(int(sys.argv[1]), int(sys.argv[2]))
