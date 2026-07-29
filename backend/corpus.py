import json
import sqlite3
import sys
import unicodedata
import os

def get_db_name():
    DB_NAME = "./manchu_transliteration.db"
    return DB_NAME

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
    cursor.execute("DROP TABLE corpus")

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
    connection = sqlite3.connect(get_db_name())
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
    connection = sqlite3.connect(get_db_name())
    cursor = connection.cursor()
    rows = cursor.execute("SELECT manchu, romanization FROM corpus ORDER BY romanization").fetchall()
    size = len(rows)
    connection.close()
    return size

def get_random_word():
    connection = sqlite3.connect(get_db_name())
    cursor = connection.cursor()
    word_set = cursor.execute("SELECT  * FROM corpus ORDER BY RANDOM() LIMIT 1;").fetchone()
    word_dict = { "manchu": word_set[0], "romanization": word_set[1] }
    connection.close()
    return word_dict

def generate_new_lexigraph():
    return

if __name__ == "__main__":
    create_corpus()
    print(get_corpus_size())
