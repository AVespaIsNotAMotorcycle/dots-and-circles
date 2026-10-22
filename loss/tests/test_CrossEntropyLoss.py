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

            expected = np.zeros(num_classes)
            for item in range(batch_size):
                for token in range(context_length):
                    exp = np.exp(y[item][token])
                    probs = exp / np.sum(exp)
                    probs[label[item][token]] -= 1
                    expected += probs
            expected /= batch_size * context_length

            gradient = loss_function.loss_gradient(None, y, label)
            assert np.shape(gradient) == (1, num_classes)
            assert np.allclose(gradient[0], expected)
            # Each row of probs - one_hot sums to 0, so the mean does too
            assert np.isclose(np.sum(gradient), 0)

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
