"""
Word2Vec — Skip-Gram with Negative Sampling (SGNS), implemented from scratch.

Why two matrices (W_in, W_out)?
--------------------------------
Every word gets TWO vectors during training:
  - W_in[word]  -> its vector when the word is the INPUT (the "embedding" matrix)
  - W_out[word] -> its vector when the word is being predicted as CONTEXT

We only keep W_in at the end — that's the word embedding everyone talks about.
W_out is scaffolding we throw away once training is done.

Why negative sampling?
-----------------------
Predicting "is this the right neighbor?" over the WHOLE vocabulary (softmax) is
expensive if the vocab has 50,000+ words — you'd update every single row on every
single step. Negative sampling turns this into a much cheaper binary classification:
"is (center, context) a real neighbor pair? yes/no" — trained against a handful of
random "no" examples (negative samples) instead of the entire vocabulary.
"""

import numpy as np


class Word2Vec:
    def __init__(self, vocab_size, embedding_dim=100, learning_rate=0.025,
                 negative_samples=5, seed=42):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lr = learning_rate
        self.k = negative_samples

        rng = np.random.default_rng(seed)
        # Small random init for the embedding matrix (this is what we keep)
        self.W_in = (rng.random((vocab_size, embedding_dim)) - 0.5) / embedding_dim
        # Context matrix starts at zero (common word2vec convention)
        self.W_out = np.zeros((vocab_size, embedding_dim))

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -10, 10)  # avoid overflow in exp()
        return 1.0 / (1.0 + np.exp(-x))

    def train_pair(self, center_idx, context_idx, negative_idxs):
        """
        One training step for ONE positive pair (center, context) plus its
        negative samples. Returns the loss for monitoring.

        Math recap:
          score      = sigmoid(v_center . v_context)   -> should be close to 1 (positive pair)
          neg_scores = sigmoid(v_center . v_negative)   -> should be close to 0 (fake pairs)
          error      = prediction - target
          gradient   = error * (the other vector)
        """
        v_center = self.W_in[center_idx]                      # (d,)

        # ---- positive example: center + real context word, target = 1 ----
        v_pos = self.W_out[context_idx]                        # (d,)
        pred_pos = self.sigmoid(np.dot(v_center, v_pos))
        error_pos = pred_pos - 1.0                              # target is 1

        # ---- negative examples: center + random words, target = 0 ----
        v_neg = self.W_out[negative_idxs]                      # (k, d)
        pred_neg = self.sigmoid(v_neg @ v_center)               # (k,)
        error_neg = pred_neg                                    # target is 0

        # ---- gradient w.r.t. the center vector (sum of both contributions) ----
        grad_center = error_pos * v_pos + error_neg @ v_neg     # (d,)

        # ---- update context vectors first (before we change v_center!) ----
        self.W_out[context_idx] -= self.lr * error_pos * v_center
        self.W_out[negative_idxs] -= self.lr * np.outer(error_neg, v_center)

        # ---- update the center word's embedding ----
        self.W_in[center_idx] -= self.lr * grad_center

        loss = -np.log(pred_pos + 1e-10) - np.sum(np.log(1 - pred_neg + 1e-10))
        return loss

    def get_vector(self, idx):
        return self.W_in[idx]
