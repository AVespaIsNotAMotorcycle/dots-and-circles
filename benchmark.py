import os
import re
import random
from PIL import Image
from Levenshtein import ratio
import matplotlib.pyplot as plt

from constants import ALPHABET
from corpus import get_all_words
from lexigraphy import FONTS, create_lexigraph

from model1 import Model1
from model2 import Model2

word_count = 24973
font_count = 11
seed_range = 1000
training_data_count = word_count * font_count * seed_range
def load_train_data(max_entries = 0):
    print(f"{training_data_count:,} entries in training data.")

    data = []

    words = get_all_words()
    random.shuffle(words)
    if max_entries == 0: max_entries = len(words)
    for index, word in enumerate(words):
        if len(data) >= max_entries * len(FONTS):
            break
        for char in word:
            if char not in ALPHABET:
                continue

        manchu = word[0]
        
        progress = (index + 1) / max_entries * 100
        print(f"{progress:6.2f}% | {manchu}")

        for font in range(len(FONTS)):
            for seed in range(1):
                lexigraph = create_lexigraph(manchu, font, crop=True)
                data.append((lexigraph, manchu))

    random.shuffle(data)

    batches = []
    batch = []
    for entry in data:
        batch.append(entry)
        if len(batch) == 100:
            batches.append(batch)
            batch = []

    return batches

def load_test_data(max_entries = 0):
    directories = list(os.walk('scidb'))
    random.shuffle(directories)

    data = []
    for folder, _, files in directories:
        random.shuffle(files)
        if len(files) == 0: continue

        file = files[0]
        filename = f"{folder}/{file}"
        label = re.sub('scidb/', '', folder)
        data.append([filename, label])
        if max_entries > 0 and len(data) >= max_entries: break

    random.shuffle(data)
    return data

def load_image(filename):
    image = Image.open(filename)
    return image

def plot_train(performance):
    for i, model in enumerate(performance[0].keys()):
        x = []
        y = []
        for j, batch in enumerate(performance):
            x.append(j + 1)
            y.append(batch[model]['accuracy'])
        plt.plot(x, y, label=model)

    plt.legend(title="Accuracy per batch of 100 samples")
    plt.show()

def benchmark_train_accuracy(models):
    performance = []

    data = load_train_data(2000)

    for index, batch in enumerate(data):
        batch_performance = {}
        for model in models:
            accuracy, loss = model.train(batch)
            batch_performance[model.name()] = { 'accuracy': accuracy, 'loss': loss }
            model.save()
        if index % 5 == 0:
            print(f"Batch {index + 1} ===============================================")
            for key in batch_performance.keys():
                print((f"{key} |"
                       f" {batch_performance[key]['accuracy']:3.0%} |"
                       f" {batch_performance[key]['loss']:,.2f}"))
        performance.append(batch_performance)

    plot_train(performance)
    return performance

def benchmark_test_accuracy(models):
    performance = {}
    for model in models:
        performance[model.name()] = 0

    data = load_test_data(5)
    for entry in data:
        filename, label = entry
        image = load_image(filename)
        for model in models:
            prediction = model.predict(image, output="abkai")
            performance[model.name()] += ratio(prediction, label)

    for key in performance.keys():
        performance[key] = performance[key] / len(data)
    return performance, len(data)

model1 = Model1()
model2 = Model2()
models = [model1, model2]
performance = benchmark_train_accuracy(models)
