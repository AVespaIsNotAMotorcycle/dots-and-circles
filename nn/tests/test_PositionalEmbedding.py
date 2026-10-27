import numpy as np
import pytest

from nn.PositionalEmbedding import PositionalEmbedding


class TestPositionalEmbedding:
    def setup_method(self):
        np.random.seed(123)

    def embeddings(self, module, num_tokens, emb_dim):
        # With x = 0 the output is just the embeddings
        return module(np.zeros((1, num_tokens, emb_dim)))[0]

    def test_shape(self):
        for learned in [False, True]:
            module = PositionalEmbedding(8, context_length=16,
                                         learned=learned)
            for shape in [(5, 8), (3, 5, 8), (2, 16, 8)]:
                x = np.random.randn(*shape)
                assert np.shape(module(x)) == shape

    def test_same_across_batch(self):
        # Every item in a batch gets the same offset at a given position
        for learned in [False, True]:
            module = PositionalEmbedding(8, context_length=16,
                                         learned=learned)
            x = np.random.randn(4, 6, 8)
            offsets = module(x) - x
            for item in range(1, 4):
                assert np.allclose(offsets[item], offsets[0])

    def test_positions_differ(self):
        for learned in [False, True]:
            module = PositionalEmbedding(8, context_length=16,
                                         learned=learned)
            embeddings = self.embeddings(module, 16, 8)
            for i in range(16):
                for j in range(i + 1, 16):
                    assert not np.allclose(embeddings[i], embeddings[j])

    def test_fewer_tokens(self):
        # Shorter inputs get the first num_tokens positions
        for learned in [False, True]:
            module = PositionalEmbedding(8, context_length=16,
                                         learned=learned)
            full = self.embeddings(module, 16, 8)
            assert np.allclose(self.embeddings(module, 5, 8), full[:5])

    def test_sinusoidal_values(self):
        module = PositionalEmbedding(4)
        embeddings = self.embeddings(module, 3, 4)
        for position in range(3):
            expected = [np.sin(position),
                        np.cos(position),
                        np.sin(position / 100),
                        np.cos(position / 100)]
            assert np.allclose(embeddings[position], expected)

    def test_sinusoidal_any_length(self):
        # Fixed embeddings have no context_length limit or parameters
        module = PositionalEmbedding(7)
        assert module.parameters() == {}
        assert np.shape(module(np.zeros((1, 1000, 7)))) == (1, 1000, 7)

    def test_learned_limit(self):
        with pytest.raises(AssertionError):
            PositionalEmbedding(8, learned=True)

        module = PositionalEmbedding(8, context_length=16, learned=True)
        assert np.shape(module.parameters()['embeddings']) == (16, 8)
        with pytest.raises(AssertionError):
            module(np.zeros((1, 17, 8)))
