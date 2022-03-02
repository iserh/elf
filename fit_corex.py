"""Training the topic model."""
from pathlib import Path

import numpy as np
import scipy.sparse as ss
from corextopic import corextopic as ct
import json

N_HIDDEN = 128
MAX_ITER=30
SEED = 8
VECTORIZER_FEAT = 10_000

src_dir = Path("/home/iailab36/iser/models") / f"sts_vec={VECTORIZER_FEAT}"
assert src_dir.exists(), f"{src_dir} doesn't exist"
out_dir = Path("/home/iailab36/iser/models") / f"sts_vec={VECTORIZER_FEAT}" / f"corex_n_hidden={N_HIDDEN}_iter={MAX_ITER}"
out_dir.mkdir(exist_ok=True, parents=True)


print("Loading doc_word.")
doc_word = ss.load_npz(src_dir / "doc_word.npz")
np.random.seed(SEED)
perm = np.random.permutation(doc_word.shape[0])
doc_word = doc_word[perm][:2_000_000]
del perm

print(doc_word.shape)

print("Training topic model.")
# Train the CorEx topic model with 50 topics
topic_model = ct.Corex(n_hidden=N_HIDDEN, max_iter=MAX_ITER, seed=SEED, verbose=True)
topic_model.fit(doc_word)

print("Saving topic model.")
# save topic model
topic_model.save(out_dir / "corex_model.bin")

info = {
    "N_HIDDEN": 128,
    "MAX_ITER": 10,
    "SEED": 8,
    "VECTORIZER_FEAT": 10_000,
    "tc": topic_model.tc,
    "tcs": topic_model.tcs.tolist(),
    "tc_history": topic_model.tc_history,
}

with open(out_dir / "info.json", "w") as f:
    json.dump(info, f)
