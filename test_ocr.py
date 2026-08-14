import lexigraphy
import numpy as np

import constants
from ocr import NeuralNetwork

sizes = [16, 32, 64, 128, 256]

def test_input_format():
    manchu = 'ᠰᡳᠮᠨᡝᠪᡠᠮᠪᡳ'
    font = 5

    slices, row_labels = lexigraphy.get_slices(font, manchu)
    assert np.shape(slices[0]) == (constants.INPUT_LAYER_SIZE, 1)

def test_forward_propogate():
    manchu = 'ᠰᡳᠮᠨᡝᠪᡠᠮᠪᡳ'
    font = 5

    slices, row_labels = lexigraphy.get_slices(font, manchu)

    for hidden_layer_size in sizes:
        nn = NeuralNetwork(hidden_layer_size)
        for slice in slices:
            y = nn.forward(slice)
            assert np.shape(y) == (constants.OUTPUT_LAYER_SIZE, 1)

'''
def test_mse_loss():
    test_cases = [[np.ones(constants.OUTPUT_LAYER_SIZE),
                   20,
                   (constants.OUTPUT_LAYER_SIZE - 1) / constants.OUTPUT_LAYER_SIZE],
                  [np.zeros(constants.OUTPUT_LAYER_SIZE),
                   20,
                   1 / constants.OUTPUT_LAYER_SIZE]]
    for case in test_cases:
        output, actual_letter, expected_loss = case
        assert mse_loss(output, actual_letter) == expected_loss
'''

'''
def test_backward_propogate():
    manchu = 'ᠰᡳᠮᠨᡝᠪᡠᠮᠪᡳ'
    font = 5

    slices, row_labels = lexigraphy.get_slices(font, manchu)

    for hidden_layer_size in sizes:
        nn = NeuralNetwork(hidden_layer_size)
        for slice in slices:
            y = nn.forward_propogate(slice)
            np.shape(y) == (constants.OUTPUT_LAYER_SIZE, 1)
            assert np.shape(y) == (constants.OUTPUT_LAYER_SIZE, 1)
'''

def test_neuron_value_shape():
    for hidden_layer_size in sizes:
        nn = NeuralNetwork(hidden_layer_size)
        internals = nn.get_neurons()

        assert np.shape(internals['weight_x_h']) == (hidden_layer_size, constants.INPUT_LAYER_SIZE)
        assert np.shape(internals['weight_h_y']) == (constants.OUTPUT_LAYER_SIZE, hidden_layer_size)
        assert np.shape(internals['bias_h']) == (hidden_layer_size, 1)
        assert np.shape(internals['bias_y']) == (constants.OUTPUT_LAYER_SIZE, 1)

'''

def test_train_on_lexigraph():
    manchu = 'ᠰᡳᠮᠨᡝᠪᡠᠮᠪᡳ'
    font = 5

    slices, row_labels = lexigraphy.get_slices(font, manchu)

    nn = NeuralNetwork(50)
    predictions = nn.train_on_lexigraph(slices, row_labels)
    assert len(predictions) == len(row_labels)
'''
