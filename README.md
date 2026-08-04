# Word2Vec with Gensim

A tutorial notebook introducing Gensim's `Word2Vec` model, following the
structure of the official Gensim word2vec tutorial — with one change: the
optional pretrained-model demo uses a small, self-built dataset instead of
downloading the ~2GB Google News vectors, so the whole notebook runs
out of the box with no large downloads.

## Contents

1. Review: Bag-of-words
2. Introducing the Word2Vec model
3. Word2Vec demo on a small self-built dataset
4. Training your own model (on the Lee Evaluation Corpus, bundled with Gensim)
5. Storing and loading models
6. Training parameters (`min_count`, `vector_size`, `workers`)
7. Memory requirements
8. Evaluating (word analogies + word pair similarity)
9. Online training / resuming training
10. Training loss computation
11. Visualizing embeddings (t-SNE)

## Requirements

- Python 3
- [gensim](https://radimrehurek.com/gensim/)
- numpy
- matplotlib (for the t-SNE visualization)
- scikit-learn (for t-SNE)

Install with:

```bash
pip install gensim numpy matplotlib scikit-learn
```

## Usage

Open the notebook in Jupyter (or JupyterLab / VS Code / Google Colab) and
run the cells top to bottom:

```bash
jupyter notebook word2vec_gensim.ipynb
```

No external downloads are required — the demo dataset is generated inline,
and the training corpus (Lee Evaluation Corpus) ships with Gensim itself.

## License

Add a license of your choice (e.g. MIT) if you plan to share this repo publicly.
