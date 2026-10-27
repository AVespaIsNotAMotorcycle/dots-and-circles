import numpy as np

from nn.tests.test_gradients import check_gradients
from ocr.modelD import ModelD


def small_model(drop_rate=0.3):
    return ModelD(depth=2, emb_dim=8, context_length=5, num_heads=2,
                  num_classes=4, token_dim=6, drop_rate=drop_rate)


class TestModelD:
    def setup_method(self):
        np.random.seed(123)

    def test_output_shape(self):
        # One prediction per image token
        model = small_model()
        y = model(np.random.randn(2, 5, 6))
        assert np.shape(y) == (2, 5, 4)

    def test_training_is_random(self):
        # Each forward pass drops different elements
        model = small_model()
        x = np.random.randn(2, 5, 6)
        assert not np.allclose(model(x), model(x))

    def test_evaluation_is_deterministic(self):
        model = small_model()
        model.set_training(False)
        x = np.random.randn(2, 5, 6)
        assert np.allclose(model(x), model(x))

    def test_evaluation_matches_no_dropout(self):
        # In evaluation mode, dropout is skipped entirely, so the output is
        # the same as an identical model with drop_rate 0
        model = small_model()
        undropped = small_model(drop_rate=0)
        for key, value in model.flat_parameters().items():
            undropped.flat_parameters()[key][...] = value

        model.set_training(False)
        x = np.random.randn(2, 5, 6)
        assert np.allclose(model(x), undropped(x))

    def test_gradients(self):
        # check_gradients reseeds before each forward pass, so every pass
        # drops the same elements
        for lengths in [None, [5, 2]]:
            model = small_model()
            model.set_lengths(lengths)
            check_gradients(model, np.random.randn(2, 5, 6))
