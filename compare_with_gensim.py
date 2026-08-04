"""
Optional: train the SAME corpus with Gensim's Word2Vec and compare results
against the from-scratch implementation. This is the "theory vs. production
library" comparison — useful for sanity-checking your understanding.

Requires: pip install gensim
"""

from gensim.models import Word2Vec as GensimWord2Vec
from gensim.utils import simple_preprocess


def load_sentences(path="data/sample_corpus.txt"):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    # Gensim expects a list of tokenized sentences
    return [simple_preprocess(line) for line in text.split(".") if line.strip()]


def main():
    sentences = load_sentences()
    print(f"Training on {len(sentences)} sentences...")

    model = GensimWord2Vec(
        sentences,
        vector_size=50,
        window=4,
        min_count=1,
        sg=1,          # 1 = skip-gram (matches our from-scratch implementation)
        negative=5,    # negative sampling, same as our implementation
        epochs=60,
    )

    print("\n--- Gensim: nearest neighbors ---")
    for w in ["learning", "word", "network"]:
        if w in model.wv:
            print(f"\nMost similar to '{w}':")
            for word, score in model.wv.most_similar(w, topn=5):
                print(f"  {word:<15} {score:.3f}")

    print("\nCompare these results against `python train.py` output —")
    print("they won't match exactly (different init, different corpus splitting,")
    print("Gensim uses hierarchical softmax/optimized C code under the hood) but")
    print("the *kind* of neighbors returned should feel similar.")


if __name__ == "__main__":
    main()
