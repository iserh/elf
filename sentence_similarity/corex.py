import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as ss
from corextopic import corextopic as ct
from sklearn.feature_extraction.text import TfidfVectorizer

from sentence_similarity.data import Pipeline


def train_corex_model(pipeline: Pipeline, sentences, output_dir: Path, n_hidden=100, max_iter=300, seed=1337):
    # create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    # save pipeline config
    pipeline.config.save(output_dir)
    # preprocess sentences
    sentences = pipeline(sentences)
    # fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(strip_accents="ascii", binary=True)
    doc_word = vectorizer.fit_transform(sentences)
    doc_word = ss.csr_matrix(doc_word)
    # Get words that label the columns (needed to extract readable topics and make anchoring easier)
    words = list(np.asarray(vectorizer.get_feature_names_out()))
    # Train the CorEx topic model with 50 topics
    topic_model = ct.Corex(n_hidden=n_hidden, words=words, max_iter=max_iter, verbose=False, seed=seed)
    topic_model.fit(doc_word, words=words)
    # save vectorizer
    with open(output_dir / "vectorizer.bin", "wb") as f:
        pickle.dump(vectorizer, f)
    # save topic model
    topic_model.save(output_dir / "corex_model.bin")
    return topic_model, vectorizer
