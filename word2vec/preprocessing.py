"""
Everything needed to turn raw text into training pairs for skip-gram.
"""

import re
from collections import Counter
import numpy as np

TOKEN_RE = re.compile(r"[A-Za-z]+[\w^']*|[\w^']*[A-Za-z]+[\w^']*")


def tokenize(text):
    """Lowercase + split into word tokens, stripping punctuation."""
    return TOKEN_RE.findall(text.lower())


def build_vocab(tokens, min_count=1):
    """
    Build word<->id lookup tables, dropping rare words (min_count).
    Rare words are usually noise (typos, one-off names) and there isn't
    enough data to train a meaningful vector for them anyway.
    """
    counts = Counter(tokens)
    vocab = [w for w, c in counts.items() if c >= min_count]
    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = {i: w for w, i in word_to_id.items()}
    freqs = np.array([counts[w] for w in vocab], dtype=np.float64)
    return word_to_id, id_to_word, freqs


def subsample_tokens(token_ids, freqs, threshold=1e-3, seed=42):
    """
    Randomly drop very frequent words (e.g. "the", "is") with a probability
    proportional to their frequency. This speeds up training and stops
    high-frequency, low-information words from dominating the context windows.
    Formula is the one used in the original word2vec paper.
    """
    rng = np.random.default_rng(seed)
    total = freqs.sum()
    word_probs = freqs / total
    keep_prob = (np.sqrt(word_probs / threshold) + 1) * (threshold / word_probs)
    keep_prob = np.clip(keep_prob, 0, 1)

    kept = [t for t in token_ids if rng.random() < keep_prob[t]]
    return kept


def build_negative_sampling_table(freqs, table_size=1_000_000, power=0.75):
    """
    Pre-build a big lookup table so we can sample "negative" (fake neighbor)
    words in O(1) time. Raising frequencies to the power 0.75 (instead of
    sampling by raw frequency) softens the bias toward extremely common words
    -- this specific exponent is what the original word2vec paper found worked best.
    """
    probs = freqs ** power
    probs /= probs.sum()
    cumulative = np.cumsum(probs)
    positions = np.arange(table_size) / table_size
    table = np.searchsorted(cumulative, positions)
    return table


def sample_negatives(table, k, exclude_idx, rng):
    """Draw k negative sample ids from the table, avoiding the true context word."""
    negs = []
    while len(negs) < k:
        candidate = table[rng.integers(0, len(table))]
        if candidate != exclude_idx:
            negs.append(candidate)
    return np.array(negs)


def generate_pairs(token_ids, window_size, rng):
    """
    Slide a window across the token sequence and yield (center, context) pairs.
    Window size is randomized per-center-word (1..window_size), which is the
    standard word2vec trick: it implicitly weights closer words more heavily,
    since they're included in the window more often across the corpus.
    """
    pairs = []
    n = len(token_ids)
    for i, center in enumerate(token_ids):
        dyn_window = rng.integers(1, window_size + 1)
        start = max(0, i - dyn_window)
        end = min(n, i + dyn_window + 1)
        for j in range(start, end):
            if j != i:
                pairs.append((center, token_ids[j]))
    return pairs
