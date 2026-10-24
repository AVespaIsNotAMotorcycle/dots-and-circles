import numpy as np
import pytest

from nn.Linear import Linear
from nn.Sequential import Sequential
from ocr.modelC import ModelC


def small_model():
    return ModelC(depth=2, emb_dim=8, context_length=6, num_heads=2,
                  num_classes=5, token_dim=4)


class TestWeights:
    def setup_method(self):
        np.random.seed(123)

    def test_flat_parameters_paths(self):
        model = Sequential([Linear(3, 4), Linear(4, 2)])
        params = model.flat_parameters()
        assert set(params) == {"0_Linear.weights", "0_Linear.bias",
                               "1_Linear.weights", "1_Linear.bias"}
        # Live arrays, not copies
        assert params["0_Linear.weights"] is \
            model.params["0_Linear"].params["weights"]

    def test_save_load_round_trip(self, tmp_path):
        path = tmp_path / "weights.npz"
        x = np.random.rand(2, 6, 4)

        model = small_model()
        expected = model(x)
        model.save_weights(path)

        # A fresh model has different random weights until loaded
        loaded = small_model()
        assert not np.allclose(loaded(x), expected)
        loaded.load_weights(path)
        assert np.allclose(loaded(x), expected)

    def test_load_keeps_arrays(self, tmp_path):
        # Loading copies into the existing arrays, so anything holding a
        # reference to a param sees the new values
        path = tmp_path / "weights.npz"
        small_model().save_weights(path)

        model = small_model()
        before = model.flat_parameters()
        model.load_weights(path)
        after = model.flat_parameters()
        for key in before:
            assert before[key] is after[key]

    def test_load_mismatched_model(self, tmp_path):
        path = tmp_path / "weights.npz"
        small_model().save_weights(path)

        deeper = ModelC(depth=3, emb_dim=8, context_length=6, num_heads=2,
                        num_classes=5, token_dim=4)
        with pytest.raises(AssertionError):
            deeper.load_weights(path)

        wider = ModelC(depth=2, emb_dim=16, context_length=6, num_heads=2,
                       num_classes=5, token_dim=4)
        with pytest.raises(AssertionError):
            wider.load_weights(path)
