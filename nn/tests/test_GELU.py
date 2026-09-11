import numpy as np

from nn.GELU import GELU

class TestGELU:
    def test_one(self):
        gelu = GELU(1, 1)
        pairs = [(-5, -2.2918e-7),
                 (-1, -0.158808),
                 (-0.752461, -0.170041),
                 (0, 0),
                 (1, 0.841192),
                 (5, 5)]
        for x, y in pairs:
            x = np.ones((1, 1)) * x
            y = np.ones((1, 1)) * y
            out = gelu(x)
            print(y, out)
            assert np.isclose(out, y)
