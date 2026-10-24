import copy

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

    def can_skip(self, mod_label):
        '''
        can_skip[s] is True when a path may jump from state s-2 straight to
        state s, skipping the blank at s-1. That's allowed when:
          - s >= 2 (there is a state two to the left), and
          - l'[s] is a letter, not a blank (blanks can't be skipped *to*;
            the blank at s-1 is the only thing being skipped over), and
          - l'[s] != l'[s-2], i.e. the two letters differ. For a doubled
            letter like "AA", skipping the blank between them would merge
            them into a single A, so the blank is mandatory.

        This only depends on the label, not on time, so it's worked out
        once instead of being re-checked for every (t, s) cell.

        Example, label [A, B, B] -> l' = [_, A, _, B, _, B, _]:
            s:        0  1  2  3  4  5  6
            l'[s]:    _  A  _  B  _  B  _
            can_skip: F  F  F  T  F  F  F
        (s=1 has no s-2; s=3 can skip A->B; s=5 can't, since B == B.)
        '''
        label = np.array(mod_label)
        can_skip = np.zeros(len(label), dtype=bool)
        # label[2:] lines up each state with label[:-2], the state two to
        # its left
        can_skip[2:] = (label[2:] != self.blank) & (label[2:] != label[:-2])
        return can_skip

    def forward(self, y, mod_label):
        '''
        Returns log(alpha), shape (times, states).

        alpha[t, s] is the probability of all path prefixes over tokens
        0..t that end in state s, including y's probability at (t, s).

        The recurrence is:
            alpha[t, s] = logaddexp(alpha[t-1, s],      # stay on s
                                    alpha[t-1, s-1],    # step from s-1
                                    alpha[t-1, s-2])    # skip from s-2,
                                                        #   if can_skip[s]
                          + log y[t, l'[s]]

        Every cell in row t only reads row t-1, never other cells in row t.
        So instead of looping over s, a whole row is computed at once from
        shifted copies of the previous row. Only the loop over t remains,
        since row t needs row t-1 to be finished first.
        '''
        log_probs = log_softmax(y)

        states = len(mod_label)
        times = len(y)

        # emit[t, s] = log y[t, l'[s]]: the log probability of emitting
        # state s's character at token t. Indexing the columns with the
        # whole label picks out every state's column at once; blanks (and
        # repeated letters) just pick the same column more than once.
        emit = log_probs[:, mod_label]
        can_skip = self.can_skip(mod_label)

        # -inf is log(0): states not (yet) reachable. Paths must start on
        # the leading blank (s=0) or the first letter (s=1), so only those
        # two are reachable at t=0.
        alpha = np.full((times, states), -np.inf)
        alpha[0, :2] = emit[0, :2]

        for t in range(1, times):
            # stay[s] = alpha[t-1, s]
            stay = alpha[t-1]

            # step[s] = alpha[t-1, s-1]: shift the row one place right.
            # State 0 has nothing to its left, so it gets -inf (log 0).
            #   stay = [a0  a1  a2  a3 ...]
            #   step = [-inf a0  a1  a2 ...]
            step = np.concatenate(([-np.inf], stay[:-1]))

            # skip[s] = alpha[t-1, s-2]: shift two places right, then
            # block it (-inf) wherever skipping isn't allowed.
            #   skip = [-inf -inf a0  a1 ...]
            skip = np.concatenate(([-np.inf, -np.inf], stay[:-2]))
            skip = np.where(can_skip, skip, -np.inf)

            # Sum the three ways in (logaddexp is + in log space), then
            # multiply by the emission probability (+ in log space).
            # np.logaddexp(-inf, -inf) is -inf, so unreachable states stay
            # unreachable without any special cases.
            alpha[t] = np.logaddexp(np.logaddexp(stay, step), skip) + emit[t]

        return alpha

    def backward(self, y, mod_label):
        '''
        Returns log(beta), shape (times, states).

        beta[t, s] is the probability of all path suffixes over tokens
        t..T-1 that start in state s, including y's probability at (t, s).
        (So alpha and beta both include it; see gradient_per_image.)

        This is forward() mirrored: paths now arrive from the right, and
        the loop runs backwards in time:
            beta[t, s] = logaddexp(beta[t+1, s],        # stay on s
                                   beta[t+1, s+1],      # step to s+1
                                   beta[t+1, s+2])      # skip to s+2,
                                                        #   if allowed
                         + log y[t, l'[s]]
        '''
        log_probs = log_softmax(y)

        states = len(mod_label)
        times = len(y)

        emit = log_probs[:, mod_label]

        # Moving s -> s+2 is the same jump that forward() checks at state
        # s+2, so it's allowed exactly when can_skip[s+2] is. Shift the
        # mask two places left; the last two states have no s+2.
        #   can_skip       = [c0 c1 c2 c3 c4 ...]
        #   can_skip_ahead = [c2 c3 c4 ... False False]
        can_skip_ahead = np.concatenate((self.can_skip(mod_label)[2:],
                                         [False, False]))

        # Paths must end on the last letter (s=-2) or the trailing blank
        # (s=-1), so only those two have a suffix at the last token.
        beta = np.full((times, states), -np.inf)
        beta[-1, -2:] = emit[-1, -2:]

        # t runs from times-2 down to 0
        for t in range(times - 2, -1, -1):
            # stay[s] = beta[t+1, s]
            stay = beta[t+1]

            # step[s] = beta[t+1, s+1]: shift the row one place left. The
            # last state has nothing to its right, so it gets -inf.
            #   stay = [... b3  b4  b5  b6]
            #   step = [... b4  b5  b6  -inf]
            step = np.concatenate((stay[1:], [-np.inf]))

            # skip[s] = beta[t+1, s+2]: shift two places left, then block
            # it wherever skipping isn't allowed.
            #   skip = [... b5  b6  -inf -inf]
            skip = np.concatenate((stay[2:], [-np.inf, -np.inf]))
            skip = np.where(can_skip_ahead, skip, -np.inf)

            beta[t] = np.logaddexp(np.logaddexp(stay, step), skip) + emit[t]

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
