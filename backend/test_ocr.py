import lexigraphy
import numpy as np
from ocr import NeuralNetwork, INPUT_LAYER_SIZE

def test_input_format():
    manchu = 'ᠰᡳᠮᠨᡝᠪᡠᠮᠪᡳ'
    font = 5

    slices, row_labels = lexigraphy.get_slices(font, manchu)
    assert np.shape(slices[0]) == (1, INPUT_LAYER_SIZE)

def test_forward_propogate():
    manchu = 'ᠰᡳᠮᠨᡝᠪᡠᠮᠪᡳ'
    font = 5

    slices, row_labels = lexigraphy.get_slices(font, manchu)

    nn = NeuralNetwork(50)
    for slice in slices:
        nn.forward_propogate(slice)
