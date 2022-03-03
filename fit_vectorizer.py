"""Training the topic model."""
import pickle
from pathlib import Path

import pandas as pd
import scipy.sparse as ss
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from utils import preprocess

MAX_FEATURES = 10_000

src_dir = Path("/home/iailab36/iser/data/stsbenchmark")
assert src_dir.exists()
out_dir = Path("/home/iailab36/iser/models") / f"sts_vec={MAX_FEATURES}"
out_dir.mkdir(exist_ok=True, parents=True)

df_train = pd.read_feather(src_dir / "sts-train.feather")
df_val = pd.read_feather(src_dir / "sts-dev.feather")
df = pd.concat([df_train, df_val])
docs = pd.concat([df.s1, df.s2])
# docs = df.doc

del df

# preprocessing
tqdm.pandas()
docs = docs.progress_apply(preprocess)

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
