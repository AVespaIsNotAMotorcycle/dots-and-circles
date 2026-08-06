import json
import sqlite3
import sys
import unicodedata
import os
from Levenshtein import ratio

import constants

def read_manchu_cake_db():
    file = open("./manchucake_db.json", "r", encoding="utf-8")
    entries = file.readlines()
    file.close()
    return entries

def all_manchu_characters(text):
    for character in text:
        if "MONGOLIAN" not in unicodedata.name(character): return False
    return True

def all_latin_characters(text):
    for character in text:
        if "LATIN" in unicodedata.name(character): continue
        if "COMMERCIAL AT" in unicodedata.name(character): continue
        if "APOSTROPHE" in unicodedata.name(character): continue
        if "HYPHEN-MINUS" in unicodedata.name(character): continue
        if "COMMA" in unicodedata.name(character): continue
        if "FULL STOP" in unicodedata.name(character): continue
        return False
    return True

def characters_in_corpus():
    connection = sqlite3.connect(constants.DB_NAME)
    cursor = connection.cursor()

    all_characters = []
    rows = cursor.execute("SELECT manchu, romanization FROM corpus ORDER BY romanization")
    for row in rows:
        characters = list(row[0])
        for character in characters: all_characters.append(character)
    unique_characters = set(all_characters)
    print(unique_characters)

    connection.close()

def add_entry_to_corpus(entry_string, cursor):
    entry = json.loads(entry_string)
    manchu = entry["m"].split()
    romanization = entry["r"].split()

    if len(manchu) != len(romanization): return

    data = []
    for index, word in enumerate(manchu):
        if not all_manchu_characters(manchu[index]): continue
        if not all_latin_characters(romanization[index]): continue
        data.append((manchu[index], romanization[index]))
    for item in data:
        try:
            cursor.executemany("INSERT INTO corpus VALUES (?, ?)", [item])
        except sqlite3.IntegrityError:
            continue
    return

def drop_table(cursor):
    try: cursor.execute("DROP TABLE corpus")
    except: return

def create_table(cursor):
    cursor.execute("CREATE TABLE corpus(manchu text NOT NULL PRIMARY KEY UNIQUE, romanization text)")

def print_table(cursor):
    rows = cursor.execute("SELECT manchu, romanization FROM corpus ORDER BY romanization")
    for row in rows: print(row)

def print_percent(index, entries):
    percent = str((index + 1) / len(entries) * 100)[:5].ljust(5, '0')
    message = "Adding words to corpus - {0}% done...".format(percent)
    sys.stdout.write('%s\r' % message)

def create_corpus():
    connection = sqlite3.connect(constants.DB_NAME)
    cursor = connection.cursor()

    drop_table(cursor)
    create_table(cursor)

    entries = read_manchu_cake_db()
    for index, entry in enumerate(entries):
        print_percent(index, entries)
        add_entry_to_corpus(entry, cursor)

    print_table(cursor)
    connection.commit()
    connection.close()

def get_corpus_size():
    connection = sqlite3.connect(constants.DB_NAME)
    cursor = connection.cursor()
    rows = cursor.execute("SELECT manchu, romanization FROM corpus ORDER BY romanization").fetchall()
    size = len(rows)
    connection.close()
    return size

def get_random_word():
    connection = sqlite3.connect(constants.DB_NAME)
    cursor = connection.cursor()
    word_set = cursor.execute("SELECT  * FROM corpus ORDER BY RANDOM() LIMIT 1;").fetchone()
    word_dict = { "manchu": word_set[0], "romanization": word_set[1] }
    connection.close()
    return word_dict

def generate_new_lexigraph():
    return

def autocorrect(word):
    connection = sqlite3.connect(constants.DB_NAME)
    cursor = connection.cursor()
    rows = cursor.execute("SELECT manchu FROM corpus ORDER BY manchu").fetchall()

    best_index = 0
    best_similarity = 0
    for index, row in enumerate(rows):
        manchu = row[0]
        similarity = ratio(word, manchu)
        if similarity > best_similarity:
            best_index = index
            best_similarity = similarity

    connection.close()
    return { "word": rows[best_index][0], "similarity": best_similarity }

if __name__ == "__main__":
    # create_corpus()
    # print(get_corpus_size())
    characters_in_corpus()
