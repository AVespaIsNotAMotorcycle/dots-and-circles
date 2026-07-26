import json

def read_manchu_cake_db():
    file = open("db.json", "r", encoding="utf-8")
    entries = file.readlines()
    file.close()
    return entries

def add_entry_to_corpus(entry_string):
    entry = json.loads(entry_string)
    manchu = entry["m"].split()
    romanization = entry["r"].split()

    if len(manchu) != len(romanization): return

    print('===============\n', manchu, '\n', romanization, '\n')
    return

def create_corpus():
    entries = read_manchu_cake_db()
    for entry in entries: add_entry_to_corpus(entry)
    return

def get_corpus_size():
    return 0

def get_random_word():
    return

if __name__ == "__main__":
    create_corpus()
