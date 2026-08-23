import os
import re
import random
from PIL import Image

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
        if max_entries > 0 and len(data) > max_entries: break

    random.shuffle(data)
    return data

def load_image(filename):
    image = Image.open(filename)
    return image

def benchmark_test_accuracy(models):
    performance = {}
    for model in models:
        performance[model.name()] = 0

    data = load_test_data()
    for entry in data:
        filename, label = entry
        image = load_image(filename)
        for model in models:
            prediction = model.predict(image, output="abkai")
            correct = prediction == label
            performance[model.name()] += int(correct)

    return performance, len(data)

print(benchmark_test_accuracy([]))
