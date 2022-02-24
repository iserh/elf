"""Training the topic model."""
import pickle
from pathlib import Path

import pandas as pd
import scipy.sparse as ss
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from utils import preprocess_factory

MAX_FEATURES = 10_000

src_dir = Path("/home/iailab36/iser/data/stsbenchmark")
assert src_dir.exists()
out_dir = Path("/home/iailab36/iser/models") / f"vectorizer-sts-{MAX_FEATURES}"
out_dir.mkdir(exist_ok=True, parents=True)

df = pd.read_feather(src_dir / "sts-train.feather")
docs = pd.concat([df.s1, df.s2])

del df

# preprocessing ------------
preprocess = preprocess_factory()
tqdm.pandas()
docs = docs.progress_apply(preprocess)
# from sentence_similarity.data.preprocess import PipelineConfig, Pipeline
# config = PipelineConfig()
# pipeline = Pipeline(config)
# docs = pipeline(docs)
# --------------------------

print("creating vectorizer.")

vectorizer = TfidfVectorizer(strip_accents="ascii", max_features=MAX_FEATURES, lowercase=True, binary=True)
doc_word = vectorizer.fit_transform(docs)
doc_word = ss.csr_matrix(doc_word)

print("doc_word:", doc_word.shape)

print("saving vectorizer.")

# save vectorizer
with open(out_dir / "vectorizer.bin", "wb") as f:
    pickle.dump(vectorizer, f)

print("saving doc_word.")

# save vectorizer
ss.save_npz(out_dir / "doc_word.npz", doc_word)

print("done.")
