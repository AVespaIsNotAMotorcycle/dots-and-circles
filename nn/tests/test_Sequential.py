import numpy as np
import json

from nn.Sequential import Sequential
from nn.Linear import Linear
from nn.GELU import GELU

class TestSequential():
    def test_forward(self):
        size_in = 50
        size_h_1 = 100
        size_h_2 = 200
        size_h_3 = 100
        size_out = 5
        x = np.ones((1, size_in))

        np.random.seed(123)
        l1 = Linear(size_in, size_h_1)
        l2 = Linear(size_h_1, size_h_2)
        l3 = Linear(size_h_2, size_h_3)
        l4 = Linear(size_h_3, size_out)

        gelu = GELU()

        seq = Sequential([l1,
                          gelu,
                          l2,
                          gelu,
                          l3,
                          gelu,
                          l4])
        y1 = seq(x)

        y2 = l1(x)
        y2 = gelu(y2)
        y2 = l2(y2)
        y2 = gelu(y2)
        y2 = l3(y2)
        y2 = gelu(y2)
        y2 = l4(y2)

        assert np.array_equal(y1, y2)

    def test_parameters(self):
        size_in = 50
        size_h_1 = 100
        size_h_2 = 200
        size_h_3 = 100
        size_out = 5
        x = np.ones((1, size_in))

        np.random.seed(123)
        l1 = Linear(size_in, size_h_1)
        l2 = Linear(size_h_1, size_h_2)
        l3 = Linear(size_h_2, size_h_3)
        l4 = Linear(size_h_3, size_out)

        gelu = GELU()

        seq = Sequential([l1,
                          gelu,
                          l2,
                          gelu,
                          l3,
                          gelu,
                          l4])

        params = seq.parameters(only_shape=True)
        # string = json.dumps(params, indent=4)
        # print(string)
        assert len(params.items()) == 7
