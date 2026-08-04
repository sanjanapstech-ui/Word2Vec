# Word2Vec

A NumPy-only implementation of **Word2Vec (Skip-Gram with Negative Sampling)** — no
TensorFlow, no PyTorch, no Gensim required to train. Built to actually understand
how word embeddings work, not just call a library function.

A comparison script against Gensim's production implementation is included too.

## What is this, in plain words?

Word2Vec turns words into lists of numbers (vectors) such that words with similar
*meaning* end up with similar *vectors*. It learns this purely by looking at which
words tend to sit next to each other in a huge amount of text — no dictionary, no
hand-labeled data needed.

The famous party trick this enables:

```
vector("king") - vector("man") + vector("woman") ≈ vector("queen")
```

## How it actually works (the short version)

1. **Tokenize** the text into words.
2. **Slide a window** across the text. For every word (the "center" word), grab
   the words around it (the "context" words). These become `(center, context)`
   training pairs — e.g. in "the cat sat on the mat", `(sat, cat)` is a pair.
3. **Negative sampling**: for every real pair, also generate a few *fake* pairs
   by pairing the center word with random words from the vocabulary. The model's
   job becomes: "is this a real neighbor, or a fake one?" — a fast binary
   classification instead of predicting across the entire vocabulary.
4. **Train**: for each pair, the model computes how wrong its guess was and
   nudges two matrices (`W_in` for input/center words, `W_out` for
   context words) slightly in the right direction.
5. **Keep `W_in`** once training is done — the row for each word IS that word's
   embedding.

Why this works: words that show up in similar contexts (e.g. "king" and "queen"
both often appear near "throne", "crown", "royal") get pushed toward similar
vectors, because they're trained on similar (center, context) pairs.

## Project structure

```
word2vec-from-scratch/
├── word2vec/
│   ├── model.py          # Word2Vec class: the actual skip-gram + negative sampling math
│   ├── preprocessing.py  # tokenizer, vocab builder, subsampling, negative sample table
│   └── utils.py          # most_similar(), analogy() — query a trained model
├── data/
│   └── sample_corpus.txt # small demo corpus (swap in your own text file)
├── train.py               # end-to-end training script
├── compare_with_gensim.py # optional: compare against Gensim's implementation
└── requirements.txt
```

## Setup

```bash
git clone <your-repo-url>
cd word2vec-from-scratch
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

## Train it

```bash
python train.py
```

With custom settings:

```bash
python train.py --corpus data/sample_corpus.txt --dim 100 --window 5 --epochs 100
```

This will:
- tokenize and build a vocabulary from your corpus
- subsample overly-frequent words (like "the", "is")
- train the skip-gram model with negative sampling
- save embeddings to `output/embeddings.npy` and `output/word_to_id.json`
- print nearest-neighbor and analogy demos

## Use the trained embeddings

```python
import json, numpy as np
from word2vec import Word2Vec, most_similar, analogy

embeddings = np.load("output/embeddings.npy")
word_to_id = json.load(open("output/word_to_id.json"))
id_to_word = {i: w for w, i in word_to_id.items()}

model = Word2Vec(vocab_size=len(word_to_id), embedding_dim=embeddings.shape[1])
model.W_in = embeddings

print(most_similar("learning", model, word_to_id, id_to_word))
```

## Notes / honest limitations

- The included `sample_corpus.txt` is tiny (a few paragraphs) — great for
  verifying the code runs and produces sane-looking neighbors, but **too small
  for genuinely meaningful embeddings** (that needs millions of words, like
  Wikipedia dumps or the text8 corpus). Swap in a bigger `.txt` file via
  `--corpus` once you've confirmed everything works.
- This is a from-scratch educational implementation, prioritizing readability
  over speed. Gensim's C-optimized implementation is orders of magnitude
  faster — `compare_with_gensim.py` shows how the "real" library version looks.

## What each hyperparameter actually does

| Parameter | What it controls |
|---|---|
| `--dim` | Length of each word's vector. Bigger = more expressive, needs more data. |
| `--window` | How many neighboring words count as "context". Small windows (2–5) favor *interchangeable* words (good/bad); large windows (15+) favor *related* words. |
| `--negatives` | How many fake pairs to contrast against each real pair. 5–20 is typical; more helps with small datasets. |
| `--epochs` | How many passes over the corpus. |
| `--min_count` | Drop words appearing fewer than this many times (removes noise/typos). |

## References

- Mikolov et al., [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781) (the original word2vec paper)
- Mikolov et al., [Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) (introduces negative sampling)
- Jay Alammar, [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/)
