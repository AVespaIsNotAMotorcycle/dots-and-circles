import random
import json
import numpy as np
from numpy.random import randn

import utils
from preprocessing import image_string_to_marks
from rnn import RNN

def getTrainingData(start = 0, end = -1):
    file = open("training_data.json", "r")
    raw_lines = file.readlines()
    lines = []
    if end == -1: end = len(raw_lines)
    for index in range(start, end):
        try:
            line = raw_lines[index]
            lines.append(json.loads(line))
        except: continue
    file.close()
    return lines

def formatImage(image_string):
    return image_string_to_marks(image_string)

def wordToArray(input_word):
    word = input_word.strip().ljust(utils.WORD_MAX_CHARACTERS, ' ')
    array = [0] * utils.OUTPUT_LAYER_SIZE
    for i in range(len(word)):
        char = word[i]
        if char not in utils.ALPHABET: print("[{0}] {1}".format(char, unicodedata.name(char)))
        char_index = utils.ALPHABET.index(char)
        array_index = (i * utils.CHARACTERS_IN_ALPHABET) + char_index
        array[array_index] = 1
    return array

def arrayToWord(array):
    word = ""
    for char_index in range(utils.WORD_MAX_CHARACTERS):
        start_index = char_index * utils.CHARACTERS_IN_ALPHABET
        end_index = start_index + utils.CHARACTERS_IN_ALPHABET

        max_certainty = 0
        max_index = start_index
        for i in range(start_index, end_index):
            certainty = array[i][0]
            if certainty > max_certainty:
                max_certainty = certainty
                max_index = i
        letter_index = max_index - start_index
        letter = utils.ALPHABET[letter_index]
        word += letter
    return word.strip()

def softmax(array):
    return np.exp(array) / sum(np.exp(array))

def outputToProbs(array):
    probs = []
    for row_index in range(utils.WORD_MAX_CHARACTERS):
        start = row_index * utils.CHARACTERS_IN_ALPHABET
        end = start + utils.CHARACTERS_IN_ALPHABET
        row = array[start:end]
        row_probs = softmax(row)
        for item in row_probs: probs.append(item)
    return probs

def crossEntropyLoss(output, target):
    lossPerLetter = []
    for char_index in range(utils.WORD_MAX_CHARACTERS):
        start = char_index * utils.CHARACTERS_IN_ALPHABET
        end = start + utils.CHARACTERS_IN_ALPHABET

        output_slice = output[start:end]
        target_slice = target[start:end]
        index = target_slice.index(1)
        probability = output_slice[index]
        loss = np.log(probability) * -1
        lossPerLetter.append(loss)

    return sum(lossPerLetter) / len(lossPerLetter)

def probsToWord(probs):
    maxxed = []
    for char_index in range(utils.WORD_MAX_CHARACTERS):
        start = char_index * utils.CHARACTERS_IN_ALPHABET
        end = start + utils.CHARACTERS_IN_ALPHABET
        max_index = np.argmax(probs[start:end])

        row = [0] * utils.CHARACTERS_IN_ALPHABET
        row[max_index] = 1

        for char in row: maxxed.append(char)
    return maxxed

rnn = RNN(utils.INPUT_LAYER_SIZE, utils.OUTPUT_LAYER_SIZE, 256)
logs = []

def processData(data, backprop=True, print_logs=False):
    # data is an array of dictionaries mapping images to strings
    # backprop determines if the backward phase should be run.

    if len(data) == 0: return 0, 0

    loss = 0
    num_correct = 0

    for entry in data:
        x = entry['image']
        y = entry['word']
        inputs = formatImage(x)
        target = wordToArray(y)

        # Forward
        out, _ = rnn.forward(inputs)
        # probs = softmax(out)
        probs = outputToProbs(out.reshape(np.shape(out)[0]))

        # Calculate loss / accuracy
        loss += crossEntropyLoss(probs, target)
        is_correct = probsToWord(probs) == target
        num_correct += int(is_correct)

        logs.append({ "actual": y, "predicted": arrayToWord(out), "correct": is_correct })

        if print_logs:
            is_correct_string = '\033[92mRIGHT\033[0m' if is_correct else '\033[91mWRONG\033[0m'
            print('{0} | {1} => {2}'.format(is_correct_string, y, arrayToWord(out)))

        if  backprop:
            # Build dL/dy
            d_L_d_y = np.array(probs)
            d_L_d_y -= np.array(target)
            d_L_d_y = np.reshape(d_L_d_y, (utils.OUTPUT_LAYER_SIZE, 1))

            # Backward
            rnn.backprop(d_L_d_y)

    return loss / len(data), num_correct / len(data)

# Training loop

data = getTrainingData()
random.shuffle(data)
for epoch in range(1000):
    BATCH_SIZE = 50
    TEST_SIZE = 0

    start = epoch * BATCH_SIZE
    division = start + BATCH_SIZE - TEST_SIZE
    end = division + TEST_SIZE

    train_data = data[start:division]
    test_data = data[division:end]
    train_loss, train_acc = processData(train_data, True, True)

    if epoch % 100 == 99:
        print('--- Epoch %d' % (epoch + 1))
        print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss, train_acc))
        
        test_loss, test_acc = processData(test_data, backprop=False, print_logs=True)
        print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss, test_acc))

file = open("training_logs.json", "w")
for entry in logs:
    file.write(json.dumps(entry))
    file.write('\r')
file.close()
