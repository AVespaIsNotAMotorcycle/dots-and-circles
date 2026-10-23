import random

from modelA import ModelA

FONT_COUNT = 11
WORD_COUNT = 1989

def generate_seeds(validation_set = False):
    train_proportion = 0.7
    valid_proportion = 0
    test_proportion = 0.3

    if validation_set:
        valid_proportion = 0.15
        test_proportion = 0.15

    train_set = []
    valid_set = []
    test_set = []

    seed_count = FONT_COUNT * WORD_COUNT
    for i in range(seed_count):
        seed = i
        lot = random.randrange(1)
        if lot < train_proportion:
            train_set.append(seed)
        elif lot < train_proportion + valid_proportion:
            valid_set.append(seed)
        else:
            test_set.append(seed)

    random.shuffle(train_set)
    random.shuffle(valid_set)
    random.shuffle(test_set)

    if validation_set:
        return train_set, valid_set, test_set

    return train_set, test_set

def word_to_array(word):
    '''
    Converts a word to an array of one-hot vectors. Each vector represents
    a character in that word. Each vector is of length n+1, where n is the
    number of characters in the alphabet (see training_data/alphabet.json)
    and the +1 corresponds to an extra character which indicates the end of
    the word. The array should always end with that character's one-hot
    vector, and that vector should never appear except as the final element
    of the array.
    '''
    raise NotImplementedError

def array_to_word(array):
    '''
    Undoes word_to_array.
    '''
    raise NotImplementedError

def generate_batches(seeds, batch_size=64):
    '''
    Converts seeds into words and images, and returns them as an array of
    batches.

    Batch: array of length batch_size, where each item is an Entry.
    Entry: tuple of form (Image, Label).
    Image: PIL Image converted to a grayscale array,
        made with training_data/__init__.py
    Label: An array of one-hot vectors, each one representing a character
        in the word.
    '''
    raise NotImplementedError

def train_on_batch(model, batch, train=False):
    # Step 1: Forward Propogate
    # Step 2: Calculate Accuracy and Loss
    if not train:
        return accuracy, loss

    # Step 3: Calculate Gradients
    # Step 4: Optimize
    return accuracy, loss

def log_batch(epoch, batch_number, train_accuracy, train_loss,
              test_accuracy, test_loss):
    # Print info

def graph(train_history, test_histor):
    # Graph with matplotlib

def train(epochs = 3):
    train_set, test_set = generate_seeds()

    train_set = generate_batches(train_set)
    test_set = generate_batches(test_set)
    batch_count = len(train_set) + len(test_set)

    model = ModelA()
    
    train_history = []
    test_history = []
    for epoch in range(epochs):
        train_index = 0
        test_index = 0
        for batch_number in range(batch_count):
            if batch_number % 10 > 7:
                batch = test_set[test_index]
                accuracy, loss = test_on_batch(model, batch, train=False)
                test_history.append([batch_number, accuracy, loss])
                test_index += 1
            else:
                batch = train_set[train_index]
                accuracy, loss = train_on_batch(model, batch, train=True)
                train_history.append([batch_number, accuracy, loss])
                train_index += 1
            if batch_number % 100 == 0:
                test_numbers = test_history[-1]
                _, test_accuracy, test_loss = test_numbers

                train_numbers = train_history[-1]
                _, train_accuracy, train_loss = train_numbers

                log_batch(epoch, batch_number, train_accuracy, train_loss,
                    test_accuracy, test_loss)


if __name__ == "__main__":
    train()
