import numpy as np

from nn.tests.test_gradients import check_gradients
from ocr.modelB import ModelB


def small_model():
    return ModelB(depth=1, emb_dim=8, context_length=5, num_heads=2,
                  num_classes=4, token_dim=6, num_readouts=3)


class TestModelB:
    def setup_method(self):
        np.random.seed(123)

    def test_output_shape(self):
        # One prediction per readout token, however many image tokens
        model = small_model()
        for num_tokens in [1, 3, 5]:
            y = model(np.random.randn(2, num_tokens, 6))
            assert np.shape(y) == (2, 3, 4)

    def test_lengths_include_readouts(self):
        model = small_model()
        model.set_lengths([2, 5])
        transformer = model.params['layers'].params['3_Transformer']
        assert transformer.params['attention'].lengths == [5, 8]

    def test_padding_ignored(self):
        # Changing an item's padding tokens doesn't change its predictions
        model = small_model()
        model.set_lengths([5, 2])
        x = np.random.randn(2, 5, 6)
        changed = x.copy()
        changed[1, 2:] = np.random.randn(3, 6) * 100
        assert np.allclose(model(x), model(changed))

    def test_readouts_see_image(self):
        # Changing a real image token changes the predictions
        model = small_model()
        model.set_lengths([5, 5])
        x = np.random.randn(2, 5, 6)
        changed = x.copy()
        changed[:, 4] += 1
        assert not np.allclose(model(x), model(changed))

    def test_gradients(self):
        for lengths in [None, [5, 2]]:
            model = small_model()
            model.set_lengths(lengths)
            check_gradients(model, np.random.randn(2, 5, 6))
