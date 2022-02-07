"""Training the topic model."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as ss
from corextopic import corextopic as ct
from sklearn.feature_extraction.text import TfidfVectorizer

from sentence_similarity.data.datasets import STSB
from sentence_similarity.data.preprocess import Pipeline, PipelineConfig
from utils import get_logger

logger = get_logger(__name__)

USE_CACHE = True
N_HIDDEN = 500
MAX_ITER = 300
SEED = 1337

data_dir = Path("data")
model_dir = Path("models") / "corex_nli_stsb"

# load sts-benchmark dataset
df_train = STSB(data_dir, partition="train")
# load s-nli dataset
df_nli = pd.read_csv(data_dir / "snli_1.0" / "snli_1.0_train.txt", delimiter="\t")
# series containing sentences
sentences = pd.concat([df_nli.sentence1, df_train.s1, df_train.s2])
# sentences = pd.concat([train_data.s1, train_data.s2])

if USE_CACHE:
    logger.info("Loading cached preprocessed sentences.")
    sentences = pd.read_feather(data_dir / "_tmp" / "sentences_preprocessed.feather").sentences
else:
    # configure preprocessing pipeline
    config = PipelineConfig(
        filtered_pos_tags=[],
        use_lemmas=True,
        remove_stop_words=True,
        remove_numbers=True,
        remove_symbols=True,
        remove_punctuation=True,
    )
    pipeline = Pipeline(config)

    logger.info("Preprocessing sentences.")
    # preprocess sentences
    sentences = pipeline(sentences)
    # save to disk
    logger.info("Saving pipeline config.")
    pipeline.config.save(model_dir / "pipeline.cfg")
    logger.info("Caching processed sentences.")
    (data_dir / "_tmp").mkdir(exist_ok=True)
    sentences.to_frame("sentences").reset_index().to_feather(data_dir / "_tmp" / "_preprocessed.feather")

logger.info("Creating vectorizer.")
# fit TF-IDF vectorizer
vectorizer = TfidfVectorizer(strip_accents="ascii", binary=True, max_features=30_000)
doc_word = vectorizer.fit_transform(sentences)
doc_word = ss.csr_matrix(doc_word)

# Get words that label the columns (needed to extract readable topics and make anchoring easier)
words = list(np.asarray(vectorizer.get_feature_names_out()))

logger.info("Training topic model.")
# Train the CorEx topic model with 50 topics
topic_model = ct.Corex(n_hidden=N_HIDDEN, words=words, max_iter=MAX_ITER, verbose=False, seed=SEED)
topic_model.fit(doc_word, words=words)

logger.info("Saving vectorizer.")
# save vectorizer
with open(model_dir / "vectorizer.bin", "wb") as f:
    pickle.dump(vectorizer, f)
logger.info("Saving topic model.")
# save topic model
topic_model.save(model_dir / "corex_model.bin")
