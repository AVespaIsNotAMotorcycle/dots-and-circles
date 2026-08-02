import lexigraphy
import numpy as np

import constants
from ocr import NeuralNetwork

def test_input_format():
    manchu = 'ᠰᡳᠮᠨᡝᠪᡠᠮᠪᡳ'
    font = 5

    slices, row_labels = lexigraphy.get_slices(font, manchu)
    assert np.shape(slices[0]) == (1, constants.INPUT_LAYER_SIZE)

def test_forward_propogate():
    manchu = 'ᠰᡳᠮᠨᡝᠪᡠᠮᠪᡳ'
    font = 5

    slices, row_labels = lexigraphy.get_slices(font, manchu)

    nn = NeuralNetwork(50)
    for slice in slices:
        result = nn.forward_propogate(slice)
        assert str(type(result)) == "<class 'dict'>"

def test_train_on_lexigraph():
    manchu = 'ᠰᡳᠮᠨᡝᠪᡠᠮᠪᡳ'
    font = 5

    slices, row_labels = lexigraphy.get_slices(font, manchu)

    nn = NeuralNetwork(50)
    predictions = nn.train_on_lexigraph(slices, row_labels)
    assert len(predictions) == len(row_labels)
