"""
End-to-end training script.

Usage:
    python train.py
    python train.py --corpus data/sample_corpus.txt --epochs 50 --dim 50
"""

import argparse
import json
import time
import numpy as np

from word2vec import (
    Word2Vec, tokenize, build_vocab, subsample_tokens,
    build_negative_sampling_table, sample_negatives, generate_pairs,
    most_similar, analogy,
)


def main():
    parser = argparse.ArgumentParser(description="Train a Word2Vec model from scratch.")
    parser.add_argument("--corpus", default="data/sample_corpus.txt")
    parser.add_argument("--dim", type=int, default=50, help="embedding dimensionality")
    parser.add_argument("--window", type=int, default=4, help="max context window size")
    parser.add_argument("--negatives", type=int, default=5, help="negative samples per pair")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--min_count", type=int, default=1)
    parser.add_argument("--out_dir", default="output")
    args = parser.parse_args()

    rng = np.random.default_rng(42)

    # 1. Load + tokenize
    with open(args.corpus, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = tokenize(text)
    print(f"Loaded corpus: {len(tokens)} tokens")

    # 2. Build vocab
    word_to_id, id_to_word, freqs = build_vocab(tokens, min_count=args.min_count)
    vocab_size = len(word_to_id)
    print(f"Vocabulary size: {vocab_size}")

    token_ids = [word_to_id[t] for t in tokens if t in word_to_id]

    # 3. Subsample frequent words
    token_ids = subsample_tokens(token_ids, freqs)
    print(f"Tokens after subsampling: {len(token_ids)}")

    # 4. Build negative sampling table
    neg_table = build_negative_sampling_table(freqs)

    # 5. Init model
    model = Word2Vec(
        vocab_size=vocab_size,
        embedding_dim=args.dim,
        learning_rate=args.lr,
        negative_samples=args.negatives,
    )

    # 6. Train
    history = []
    start = time.time()
    for epoch in range(args.epochs):
        rng.shuffle(token_ids)  # note: shuffling here is a simplification —
        # a production trainer would re-slide the window each epoch instead.
        pairs = generate_pairs(token_ids, args.window, rng)
        epoch_loss = 0.0
        for center, context in pairs:
            negs = sample_negatives(neg_table, args.negatives, context, rng)
            epoch_loss += model.train_pair(center, context, negs)
        avg_loss = epoch_loss / max(len(pairs), 1)
        history.append(avg_loss)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d}  avg_loss={avg_loss:.4f}")

    print(f"\nTraining took {time.time() - start:.1f}s")

    # 7. Save embeddings + vocab
    import os
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, "embeddings.npy"), model.W_in)
    with open(os.path.join(args.out_dir, "word_to_id.json"), "w") as f:
        json.dump(word_to_id, f)
    print(f"Saved embeddings to {args.out_dir}/")

    # 8. Quick sanity check demo
    print("\n--- Demo: nearest neighbors ---")
    for w in ["learning", "word", "network"]:
        if w in word_to_id:
            print(f"\nMost similar to '{w}':")
            for word, score in most_similar(w, model, word_to_id, id_to_word, topn=5):
                print(f"  {word:<15} {score:.3f}")

    print("\n--- Demo: analogy (king - man + woman) ---")
    try:
        for word, score in analogy("king", "man", "woman", model, word_to_id, id_to_word, topn=5):
            print(f"  {word:<15} {score:.3f}")
    except KeyError as e:
        print(f"  Skipped — {e} (the small sample corpus may not contain this word)")


if __name__ == "__main__":
    main()
