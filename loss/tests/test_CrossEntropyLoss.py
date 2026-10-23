import numpy as np

from nn.Linear import Linear
from loss.CrossEntropyLoss import CrossEntropyLoss


class TestCrossEntropyLoss:
    shapes = [(1, 1, 4),
              (1, 4, 4),
              (2, 4, 24),
              (4, 16, 32)]

    def setup_method(self):
        np.random.seed(123)

    def test_uniform(self):
        # Equal logits give every class probability 1 / num_classes
        for batch_size, context_length, num_classes in self.shapes:
            loss_function = CrossEntropyLoss(None)
            y = np.zeros((batch_size, context_length, num_classes))
            label = np.random.randint(0, num_classes,
                                      (batch_size, context_length))
            loss = loss_function.loss(None, y, label)
            assert np.isclose(loss, np.log(num_classes))

    def test_loss(self):
        for batch_size, context_length, num_classes in self.shapes:
            loss_function = CrossEntropyLoss(None)
            y = np.random.rand(batch_size, context_length, num_classes)
            label = np.random.randint(0, num_classes,
                                      (batch_size, context_length))

            expected = 0
            for item in range(batch_size):
                for token in range(context_length):
                    exp = np.exp(y[item][token])
                    prob = exp[label[item][token]] / np.sum(exp)
                    expected -= np.log(prob)
            expected /= batch_size * context_length

            loss = loss_function.loss(None, y, label)
            assert np.isclose(loss, expected)

    def test_confident(self):
        # Confidently correct output should have lower loss than wrong output
        loss_function = CrossEntropyLoss(None)
        label = np.array([[0, 1, 2]])
        correct = np.eye(3)[label] * 10
        wrong = np.eye(3)[(label + 1) % 3] * 10
        correct_loss = loss_function.loss(None, correct, label)
        wrong_loss = loss_function.loss(None, wrong, label)
        assert correct_loss < 0.001
        assert wrong_loss > correct_loss

    def test_loss_gradient(self):
        for batch_size, context_length, num_classes in self.shapes:
            loss_function = CrossEntropyLoss(None)
            y = np.random.rand(batch_size, context_length, num_classes)
            label = np.random.randint(0, num_classes,
                                      (batch_size, context_length))

            expected = np.zeros(np.shape(y))
            for item in range(batch_size):
                for token in range(context_length):
                    exp = np.exp(y[item][token])
                    probs = exp / np.sum(exp)
                    probs[label[item][token]] -= 1
                    expected[item][token] = probs
            expected /= batch_size * context_length

            gradient = loss_function.loss_gradient(None, y, label)
            assert np.shape(gradient) == np.shape(y)
            assert np.allclose(gradient, expected)
            # Each row of probs - one_hot sums to 0
            assert np.allclose(np.sum(gradient, axis=-1), 0)

    def test_ignored_labels(self):
        # Labels below 0 contribute nothing to the loss or gradient
        loss_function = CrossEntropyLoss(None)
        y = np.random.rand(2, 5, 4)
        label = np.random.randint(0, 4, (2, 5))
        label[0, 3:] = -1
        label[1, 1:] = -1
        valid = label >= 0

        loss = loss_function.loss(None, y, label)
        expected = loss_function.loss(None, y[valid], label[valid])
        assert np.isclose(loss, expected)

        gradient = loss_function.loss_gradient(None, y, label)
        assert np.shape(gradient) == np.shape(y)
        assert np.all(gradient[~valid] == 0)
        assert np.allclose(
            gradient[valid],
            loss_function.loss_gradient(None, y[valid], label[valid]))

    def test_numerical_gradient(self):
        loss_function = CrossEntropyLoss(None)
        y = np.random.randn(2, 3, 5)
        label = np.random.randint(0, 5, (2, 3))
        label[1, 2] = -1

        eps = 1e-6
        expected = np.zeros(np.shape(y))
        for index in np.ndindex(np.shape(y)):
            original = y[index]
            y[index] = original + eps
            plus = loss_function.loss(None, y, label)
            y[index] = original - eps
            minus = loss_function.loss(None, y, label)
            y[index] = original
            expected[index] = (plus - minus) / (2 * eps)

        gradient = loss_function.loss_gradient(None, y, label)
        assert np.allclose(gradient, expected, atol=1e-7)

    def test_descend(self):
        size_in = 8
        num_classes = 4
        layer = Linear(size_in, num_classes)
        loss_function = CrossEntropyLoss(layer)
        x = np.random.rand(1, size_in)
        label = np.array([2])

        for _ in range(20):
            y = layer(x)
            loss, gradient = loss_function(x, y, label)
            layer.descend({key: value * 0.1
                           for key, value in gradient.items()})

        history = loss_function.history()
        assert len(history) == 20
        assert history[-1][0] < history[0][0]
