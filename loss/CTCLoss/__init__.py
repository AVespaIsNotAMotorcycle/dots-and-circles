import numpy as np

from loss.LossFunction import LossFunction
from utils.softmax import log_softmax


class CTCLoss(LossFunction):
    '''
    Everything is in log space: products of many small probabilities would
    otherwise underflow to 0. Multiplying becomes adding logs, adding
    becomes np.logaddexp, and 0 (unreachable) becomes -inf.
    '''
    def __init__(self, module, blank):
        super().__init__(module)
        # blank is the index of the output prob corresponding
        # to a blank character.
        self.blank = blank

    def forward(self, y, mod_label):
        # Returns log(alpha)
        log_probs = log_softmax(y)

        states = len(mod_label)
        times = len(y)

        alpha = np.full((times, states), -np.inf)
        alpha[0][0] = log_probs[0][self.blank]
        alpha[0][1] = log_probs[0][mod_label[1]]
        for t in range(1, times):
            for s in range(states):
                first = alpha[t-1][s]
                second = -np.inf
                third = -np.inf

                if s > 0:
                    second = alpha[t-1][s-1]

                if s > 1:
                    if mod_label[s] != self.blank:
                        if mod_label[s] != mod_label[s-2]:
                            third = alpha[t-1][s-2]

                factor = log_probs[t][mod_label[s]]
                alpha[t][s] = np.logaddexp(np.logaddexp(first, second),
                                           third) + factor

        return alpha

    def backward(self, y, mod_label):
        # Returns log(beta)
        log_probs = log_softmax(y)

        states = len(mod_label)
        times = len(y)

        beta = np.full((times, states), -np.inf)
        beta[-1][-1] = log_probs[-1][self.blank]
        beta[-1][-2] = log_probs[-1][mod_label[-2]]
        for delta_t in range(1, times):
            t = times - delta_t - 1
            for delta_s in range(states):
                s = states - delta_s - 1

                first = beta[t+1][s]
                second = -np.inf
                third = -np.inf

                if s + 1 < states:
                    second = beta[t+1][s+1]

                if s + 2 < states:
                    if mod_label[s] != self.blank:
                        if mod_label[s] != mod_label[s+2]:
                            third = beta[t+1][s+2]

                factor = log_probs[t][mod_label[s]]

                beta[t][s] = np.logaddexp(np.logaddexp(first, second),
                                          third) + factor

        return beta

    def add_blanks(self, label):
        mod_label = [self.blank]
        for char in label:
            mod_label.append(char)
            mod_label.append(self.blank)
        return mod_label

    def loss_per_image(self, y, label, token_count):
        assert token_count >= len(label)

        # The label, with blanks inserted at the start and end, and between
        # each letter.
        mod_label = self.add_blanks(label)

        alpha = self.forward(y[:token_count], mod_label)

        # Paths end on the last letter or the trailing blank
        log_p = np.logaddexp(alpha[-1][-1], alpha[-1][-2])
        loss = -log_p / len(label)
        return loss

    def loss(self, x, y, label, token_counts):
        batch_size, context_length, emb_dim = np.shape(x)

        loss = 0
        for index in range(batch_size):
            token_count = token_counts[index]
            loss += self.loss_per_image(y[index], label[index], token_count)

        return loss / batch_size

    def gradient_per_image(self, y, label, token_count):
        '''
        dL/dy for one image's real tokens, where L = -log(p) / len(label)
        and y is the logits.

        gamma[t, k] is the probability that class k is emitted at time t,
        given that the output spells the label. The gradient is
        probs - gamma: like cross entropy's probs - one_hot, but with a soft
        target worked out from alpha and beta.
        '''
        mod_label = self.add_blanks(label)

        alpha = self.forward(y[:token_count], mod_label)
        beta = self.backward(y[:token_count], mod_label)
        log_probs = log_softmax(y[:token_count])
        log_p = np.logaddexp(alpha[-1][-1], alpha[-1][-2])

        # alpha and beta both include y's probability at (t, s), so
        # subtract its log once
        occupancy = alpha + beta - log_probs[:, mod_label]

        # Blanks (and repeated letters) appear at several states, so each
        # class collects the occupancy of every state that emits it
        gamma = np.full(np.shape(log_probs), -np.inf)
        np.logaddexp.at(gamma.T, mod_label, occupancy.T)
        gamma = np.exp(gamma - log_p)

        return (np.exp(log_probs) - gamma) / len(label)

    def loss_gradient(self, x, y, label, token_counts):
        batch_size = np.shape(y)[0]

        # Padding tokens don't affect the loss, so their gradient stays 0
        gradient = np.zeros(np.shape(y))
        for index in range(batch_size):
            token_count = token_counts[index]
            gradient[index, :token_count] = self.gradient_per_image(
                y[index], label[index], token_count)

        # loss() is the mean over the batch
        return gradient / batch_size

    def __call__(self, x, y, label, token_counts):
        loss = self.loss(x, y, label, token_counts)

        dLdy = self.loss_gradient(x, y, label, token_counts)
        self.module.backward(dLdy)
        # Modules reuse their grads dict, so copy it to keep history intact
        gradient = copy.deepcopy(self.module.gradients())

        self._history.append((loss, gradient))
        return loss, gradient
