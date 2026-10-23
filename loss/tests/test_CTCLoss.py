import itertools

import numpy as np

from loss.CTCLoss import CTCLoss

# Classes: 0 = A, 1 = B, 2 = blank
A, B, BLANK = 0, 1, 2
NUM_CLASSES = 3

# (label, number of tokens): covers both branches of the skip rule
# (identical and different neighbours), plus short and long inputs
CASES = [([A, A], 3),
         ([A, B], 3),
         ([A, A], 5),
         ([A, B, A], 6),
         ([B], 1),
         ([B, A, B], 4)]


def softmax(y):
    exp = np.exp(y - np.max(y, axis=-1, keepdims=True))
    return exp / np.sum(exp, axis=-1, keepdims=True)


def extend(label):
    # The label with blanks at the start, the end, and between each letter
    extended = [BLANK]
    for letter in label:
        extended += [letter, BLANK]
    return extended


def brute_force(probs, label):
    '''
    p(label | image) by summing every output path that collapses to label:
    merge repeats, then drop blanks. Only practical for a few tokens.
    '''
    total = 0
    for path in itertools.product(range(NUM_CLASSES), repeat=len(probs)):
        merged = [c for i, c in enumerate(path) if i == 0 or c != path[i - 1]]
        if [c for c in merged if c != BLANK] == list(label):
            total += np.prod([probs[t, c] for t, c in enumerate(path)])
    return total


def dummy_x(*shape):
    # x is the model's input, which the loss doesn't use
    return np.zeros(shape + (6,))


class TestCTCLoss:
    def setup_method(self):
        np.random.seed(123)
        self.ctc = CTCLoss(None, blank=BLANK)

    def random_logits(self, *shape):
        return np.random.randn(*shape, NUM_CLASSES)

    # forward()

    def test_forward_shape(self):
        for label, T in CASES:
            alpha = self.ctc.forward(self.random_logits(T),
                                     self.ctc.add_blanks(label))
            assert np.shape(alpha) == (T, 2 * len(label) + 1)

    def test_forward_first_row(self):
        y = self.random_logits(4)
        probs = softmax(y)
        alpha = self.ctc.forward(y, self.ctc.add_blanks([B, A]))
        assert np.isclose(alpha[0, 0], probs[0, BLANK])
        assert np.isclose(alpha[0, 1], probs[0, B])
        assert np.all(alpha[0, 2:] == 0)

    def test_forward_matches_brute_force(self):
        for label, T in CASES:
            y = self.random_logits(T)
            alpha = self.ctc.forward(y, self.ctc.add_blanks(label))
            p = alpha[-1, -1] + alpha[-1, -2]
            assert np.isclose(p, brute_force(softmax(y), label)), \
                f"label {label}, {T} tokens"

    def test_forward_single_path(self):
        # AA over 3 tokens: A _ A is the only valid path
        y = self.random_logits(3)
        probs = softmax(y)
        alpha = self.ctc.forward(y, self.ctc.add_blanks([A, A]))
        expected = probs[0, A] * probs[1, BLANK] * probs[2, A]
        assert np.isclose(alpha[-1, -1] + alpha[-1, -2], expected)

    def test_forward_too_short(self):
        # AA over 2 tokens is impossible: the As need a blank between them
        alpha = self.ctc.forward(self.random_logits(2),
                                 self.ctc.add_blanks([A, A]))
        assert alpha[-1, -1] + alpha[-1, -2] == 0

    # backward()

    def test_backward_shape(self):
        for label, T in CASES:
            beta = self.ctc.backward(self.random_logits(T),
                                     self.ctc.add_blanks(label))
            assert np.shape(beta) == (T, 2 * len(label) + 1)

    def test_backward_last_row(self):
        y = self.random_logits(4)
        probs = softmax(y)
        beta = self.ctc.backward(y, self.ctc.add_blanks([B, A]))
        assert np.isclose(beta[-1, -1], probs[-1, BLANK])
        assert np.isclose(beta[-1, -2], probs[-1, A])
        assert np.all(beta[-1, :-2] == 0)

    def test_backward_matches_brute_force(self):
        for label, T in CASES:
            y = self.random_logits(T)
            beta = self.ctc.backward(y, self.ctc.add_blanks(label))
            p = beta[0, 0] + beta[0, 1]
            assert np.isclose(p, brute_force(softmax(y), label)), \
                f"label {label}, {T} tokens"

    def test_forward_backward_agree(self):
        # alpha and beta both include probs[t, l'_s], so dividing it out once
        # gives the total probability at every time step
        for label, T in CASES:
            y = self.random_logits(T)
            probs = softmax(y)
            alpha = self.ctc.forward(y, self.ctc.add_blanks(label))
            beta = self.ctc.backward(y, self.ctc.add_blanks(label))
            p = brute_force(probs, label)
            emitted = probs[:, extend(label)]
            for t in range(T):
                total = np.sum(alpha[t] * beta[t] / emitted[t])
                assert np.isclose(total, p), \
                    f"label {label}, {T} tokens, time step {t}"

    # loss()

    def test_loss_single_image(self):
        label = [A, B, A]
        y = self.random_logits(1, 6)
        loss = self.ctc.loss(dummy_x(1, 6), y, [label], [6])
        expected = -np.log(brute_force(softmax(y[0]), label)) / len(label)
        assert np.isclose(loss, expected)

    def test_loss_batch_mean(self):
        # Mean over images of -log p / label length, each over its own tokens
        labels = [[A, B, A], [B], [A, A]]
        token_counts = [6, 2, 4]
        y = self.random_logits(3, 6)
        loss = self.ctc.loss(dummy_x(3, 6), y, labels, token_counts)

        expected = np.mean([
            -np.log(brute_force(softmax(y[i, :count]), label)) / len(label)
            for i, (label, count) in enumerate(zip(labels, token_counts))])
        assert np.isclose(loss, expected)

    def test_loss_ignores_padding(self):
        labels = [[A, B], [B]]
        token_counts = [5, 2]
        y = self.random_logits(2, 6)
        changed = y.copy()
        for i, count in enumerate(token_counts):
            changed[i, count:] = self.random_logits(6 - count) * 100

        loss = self.ctc.loss(dummy_x(2, 6), y, labels, token_counts)
        loss_changed = self.ctc.loss(dummy_x(2, 6), changed, labels,
                                     token_counts)
        assert np.isclose(loss, loss_changed)

    # loss_gradient()

    def gradient_case(self):
        labels = [[A, B], [B, B]]
        token_counts = [5, 3]
        y = self.random_logits(2, 5)
        return dummy_x(2, 5), y, labels, token_counts

    def test_gradient_shape(self):
        x, y, labels, token_counts = self.gradient_case()
        gradient = self.ctc.loss_gradient(x, y, labels, token_counts)
        assert np.shape(gradient) == np.shape(y)

    def test_gradient_numerical(self):
        x, y, labels, token_counts = self.gradient_case()

        eps = 1e-6
        expected = np.zeros(np.shape(y))
        for index in np.ndindex(np.shape(y)):
            original = y[index]
            y[index] = original + eps
            plus = self.ctc.loss(x, y, labels, token_counts)
            y[index] = original - eps
            minus = self.ctc.loss(x, y, labels, token_counts)
            y[index] = original
            expected[index] = (plus - minus) / (2 * eps)

        gradient = self.ctc.loss_gradient(x, y, labels, token_counts)
        assert np.allclose(gradient, expected, atol=1e-6)

    def test_gradient_zero_on_padding(self):
        x, y, labels, token_counts = self.gradient_case()
        gradient = self.ctc.loss_gradient(x, y, labels, token_counts)
        for i, count in enumerate(token_counts):
            assert np.all(gradient[i, count:] == 0)

    def test_gradient_rows_sum_to_zero(self):
        # Shifting every logit in a row by the same amount doesn't change
        # softmax, so each row's gradient sums to 0
        x, y, labels, token_counts = self.gradient_case()
        gradient = self.ctc.loss_gradient(x, y, labels, token_counts)
        assert np.allclose(np.sum(gradient, axis=-1), 0)
