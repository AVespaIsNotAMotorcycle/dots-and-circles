import json
import unicodedata

filename = "kaikki.org-dictionary-Manchu.jsonl"
with open(filename, "r") as file:
    lines = file.readlines()

lexicon = set()
alphabet = set()
for line in lines:
    line = json.loads(line)
    if "forms" in line.keys():
        word = line["word"]
        has_latin = False
        for letter in word:
            if "LATIN" in unicodedata.name(letter):
                has_latin = True
                break
            alphabet.add(letter)
        if not has_latin:
            lexicon.add(word)

print(f"{len(lexicon)} unique words in lexicon.")
print(f"{len(alphabet)} unique letters in alphabet.")

alphabet = sorted(list(alphabet))
with open("alphabet.json", "w", encoding="utf-8") as file:
    json.dump(alphabet, file, ensure_ascii=False, indent=4)
    file.write('\n')

lexicon = sorted(list(lexicon))
with open("lexicon.json", "w", encoding="utf-8") as file:
    json.dump(lexicon, file, ensure_ascii=False, indent=4)
    file.write('\n')
