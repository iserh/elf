import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as ss
from corextopic import corextopic as ct
from sklearn.feature_extraction.text import TfidfVectorizer

from sentence_similarity.data import (Benchmark, Pipeline, PipelineConfig,
                                      STSBenchmark)

data_dir = Path("data")
assert data_dir.exists(), "data_dir does not exist."
output_dir = Path("data")
output_dir.mkdir(exist_ok=True, parents=True)


def train_corex_model(output_dir: Path, config: PipelineConfig, benchmark: Benchmark):
    config.save(output_dir)
    pipeline = Pipeline(config)

    # preprocess sentences
    s1_preprocessed = pipeline(benchmark.s1)
    s2_preprocessed = pipeline(benchmark.s2)

    # fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(strip_accents="ascii", binary=True)
    doc_word = vectorizer.fit_transform(pd.concat([s1_preprocessed, s2_preprocessed]))
    doc_word = ss.csr_matrix(doc_word)

    # save vectorizer
    with open(output_dir / "vectorizer.bin", "wb") as f:
        pickle.dump(vectorizer, f)

    # Get words that label the columns (needed to extract readable topics and make anchoring easier)
    words = list(np.asarray(vectorizer.get_feature_names_out()))

    # Train the CorEx topic model with 50 topics
    topic_model = ct.Corex(n_hidden=50, words=words, max_iter=300, verbose=False, seed=1)
    topic_model.fit(doc_word, words=words)

    topic_model.save(output_dir / "corex_model.bin")
