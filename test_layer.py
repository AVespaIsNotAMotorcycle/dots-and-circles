import pytest
import math
from layer import Layer
import numpy as np
from numpy.random import seed

def test_correct_size():
    size_in = 4
    size_out = 6
    layer = Layer(size_in, size_out)

    # Error on input not of shape (size_in, 1)
    bad_inputs = [np.zeros((5, 1)), np.zeros((1, 5)), np.zeros((2, 2))]
    for x in bad_inputs:
        with pytest.raises(ValueError):
            layer.forward(x)

    # No error on input of shape (size_in, 1)
    good_input = np.zeros((size_in, 1))
    output = layer.forward(good_input)

    # Output is of shape (size_out, 1)
    assert np.shape(output) == (size_out, 1)

def test_set():
    size_in = 4
    size_out = 6
    layer = Layer(size_in, size_out)

    # Error on incorrect weight type (list)
    bad_weight_1 = [[0, 1]]
    with pytest.raises(ValueError):
        layer.set(weight=bad_weight_1)

    # Error on incorrect weight shape ((1, 2))
    bad_weight_2 = np.array([[0, 1]])
    with pytest.raises(AssertionError):
        layer.set(weight=bad_weight_2)

    good_weight = np.zeros((size_out, size_in))
    good_bias = np.zeros((size_out, 1))
    layer.set(weight=good_weight, bias=good_bias)

    weight, bias = layer.get()
    assert np.array_equal(weight, good_weight)
    assert np.array_equal(bias, good_bias)

def test_node_value():
    x = np.array([[0], [1]])
    weight = np.array([[0.5, 0.2]])
    bias = np.array([[0.25]])
    activation = math.tanh
    y = np.array([[activation(
                   x[0][0] * weight[0][0] +
                   x[1][0] * weight[0][1] +
                   bias[0][0])]])

    size_in = len(x)
    size_out = len(y)

    seed(123)
    layer = Layer(size_in, size_out)
    layer.set(weight=weight, bias=bias)

    out = layer.forward(x)
    assert out == y

def test_backprop():
    x = np.array([[-1], [0], [1]])
    label = np.array([[1], [0]])

    size_in = 3
    size_out = 2
    layer = Layer(size_in, size_out)
    
    weight = np.array([[0.1, 0.2, 0.3],
                        [0.5, 0.6, 0.7]])
    bias = np.array([[0.2], [0.3]])
    layer.set(weight=weight, bias=bias)

    # Check that output is what we expect
    activation = math.tanh
    y1 = activation(((-1 * 0.1) + (0 * 0.2) + (1 * 0.3)) + 0.2)
    y2 = activation(((-1 * 0.5) + (0 * 0.6) + (1 * 0.7)) + 0.3)
    expected = np.array([[y1], [y2]])

    out = layer.forward(x)

    assert np.isclose(out, expected).all()

    mse_loss_1 = sum((label - out)**2) / len(out)
    '''
    Calculate dLdy
    l(y1)  = ((1 - y1)**2 + (0 - 0.46)**2) / 2
           = (1 - 2y + y**2 + c) / 2
           = (y**2 / 2) - (2y / 2) + c
           = (1/2)y**2 - y
    dLdy_1 = y - 1

    l(y2)  = ((1 - 0.38)**2 + (0 - y2)**2) / 2
           = (c + (-y2)**2) / 2
           = c / 2 + y2**2 / 2
    dLdy_2 = y2
    '''
    # Check that loss is decreasing
    dLdy = np.array([[out[0][0] - 1], [out[1][0]]])

    learn_rate = 0.2
    layer.backprop(x, out, dLdy, learn_rate)
    out = layer.forward(x)
    mse_loss_2 = sum((label - out)**2) / len(out)

    assert mse_loss_1 > mse_loss_2
    
    # Check that loss approaches 0
    old_loss = mse_loss_2
    iteration = 0
    max_iter = 15
    while old_loss > 0.01:
        iteration += 1

        if iteration >= max_iter: break

        out = layer.forward(x)
        new_loss = sum((label - out)**2) / len(out)
        assert new_loss <= old_loss
        old_loss = new_loss

        dLdy = np.array([[out[0][0] - 1], [out[1][0]]])
        layer.backprop(x, out, dLdy, learn_rate)
    assert iteration < max_iter + 1

def test_backprop_shape():
    dimensions = [(2, 3),
                  (3, 2),
                  (2, 2),
                  (1, 7),
                  (7, 1),
                  (5, 5),
                  (11, 3)]
    # Make sure no errors are thrown
    for size_in, size_out in dimensions:
        layer = Layer(size_in, size_out)

        x = np.zeros((size_in, 1))
        out = layer.forward(x)

        dLdy = np.zeros((size_out, 1))
        layer.backprop(x, out, dLdy, 1)
