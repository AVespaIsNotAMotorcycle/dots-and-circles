import pytest
import os
import numpy as np
from numpy.random import randn

from ann import ANN

def test_init():
    dimensions = [(2, 3),
                  (3, 2),
                  (2, 2),
                  (1, 7),
                  (7, 1),
                  (5, 5),
                  (11, 3),
                  (1248, 7)]
    layer_counts = [1, 2, 4, 10]

    for size_in, size_out in dimensions:
        for num_layers in layer_counts:
            # Check that it initializes without crashing
            net = ANN(size_in, size_out, num_layers)

            x = np.ones((size_in, 1))
            expected = np.ones((size_out, 1))
            for index in range(num_layers):
                if index == 0: expected = expected * size_in
                else: expected = expected * size_out
                expected = np.tanh(expected)

            parameters = []
            for i in range(num_layers):
                layer_in = size_in if i == 0 else size_out
                layer_out = size_out
                weight = np.ones((layer_out, layer_in))
                bias = np.zeros((layer_out, 1))
                parameters.append([weight, bias])
            net.set(parameters)

            # Check that set() works
            # Not sure why np.array_equal(net_params, parameters) didn't work, maybe bc it was a 2d array?
            net_params = net.get()
            assert len(net_params) == len(parameters)
            for index in range(len(net_params)):
                assert len(net_params[index]) == len(parameters[index])
                assert np.array_equal(net_params[index][0], parameters[index][0])
                assert np.array_equal(net_params[index][1], parameters[index][1])

            y = net.forward(x)
            assert np.isclose(y, expected).all()

def test_backprop_1():
    size_in = 3
    size_out = 3
    num_layers = 5
    net = ANN(size_in, size_out, num_layers)

    x = np.array([[0], [1], [2]])
    correct = 2
    label = np.zeros((size_out, 1))
    label[correct][0] = 1
    
    prev_loss = np.inf
    for _ in range(5):
        out = net.forward(x)
        mse_loss = sum((label - out)**2) / len(out)
        assert mse_loss < prev_loss
        prev_loss = mse_loss
        dLdy = out
        dLdy[correct][0] -= 1
    
        learn_rate = 0.1
        net.backprop(dLdy, learn_rate)

def test_backprop_2():
    size_in = 50 * 350
    size_out = 37 * 35
    num_layers = 3
    net = ANN(size_in, size_out, num_layers)

    x = np.ones((size_in, 1))
    correct = 124
    label = np.zeros((size_out, 1))
    label[correct] = 1

    prev_loss = np.inf
    for _ in range(5):
        out = net.forward(x)
        mse_loss = sum((label - out)**2) / len(out)
        assert mse_loss < prev_loss
        prev_loss = mse_loss
        dLdy = out
        dLdy[correct][0] -= 1
    
        learn_rate = 0.1
        net.backprop(dLdy, learn_rate)

@pytest.fixture
def setup_save_load():
    filename = "test_ann_save_load.json"
    yield filename
    os.remove(filename)

def test_save_and_load(setup_save_load):
    filename = setup_save_load
    size_in = 1500
    size_out = 32
    num_layers = 3

    net1 = ANN(size_in, size_out, num_layers)
    net2 = ANN(size_in, size_out, num_layers)

    net1.save(filename)
    net2.load(filename)

    assert str(net1.get()) == str(net2.get())

    x = randn(size_in, 1)
    y1 = net1.forward(x)
    y2 = net1.forward(x)

    assert np.array_equal(y1, y2)
