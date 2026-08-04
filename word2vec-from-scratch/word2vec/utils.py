"""
Query utilities for a trained Word2Vec model:
  - most_similar(word)          -> nearest neighbors by cosine similarity
  - analogy(a, b, c)            -> solves "a is to b as c is to ___" (e.g. king - man + woman)
"""

import numpy as np


def _cosine_scores(model, query_vec):
    norms = np.linalg.norm(model.W_in, axis=1) + 1e-10
    query_norm = np.linalg.norm(query_vec) + 1e-10
    return (model.W_in @ query_vec) / (norms * query_norm)


def most_similar(word, model, word_to_id, id_to_word, topn=10):
    if word not in word_to_id:
        raise KeyError(f"'{word}' not in vocabulary")
    idx = word_to_id[word]
    scores = _cosine_scores(model, model.get_vector(idx))
    ranked = np.argsort(-scores)

    results = []
    for i in ranked:
        if i == idx:
            continue
        results.append((id_to_word[i], float(scores[i])))
        if len(results) >= topn:
            break
    return results


def analogy(a, b, c, model, word_to_id, id_to_word, topn=5):
    """Solves: a - b + c = ?   e.g. analogy("king", "man", "woman") -> queen"""
    for w in (a, b, c):
        if w not in word_to_id:
            raise KeyError(f"'{w}' not in vocabulary")

    query_vec = (model.get_vector(word_to_id[a])
                 - model.get_vector(word_to_id[b])
                 + model.get_vector(word_to_id[c]))
    scores = _cosine_scores(model, query_vec)
    ranked = np.argsort(-scores)
    exclude = {word_to_id[a], word_to_id[b], word_to_id[c]}

    results = []
    for i in ranked:
        if i in exclude:
            continue
        results.append((id_to_word[i], float(scores[i])))
        if len(results) >= topn:
            break
    return results
