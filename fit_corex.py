"""Training the topic model."""
from pathlib import Path

import numpy as np
import scipy.sparse as ss
from corextopic import corextopic as ct

N_HIDDEN = 300
MAX_ITER=6
SEED = 8
VECTORIZER_FEAT = 10_000

src_dir = Path("/home/iailab36/iser/models") / f"vectorizer-sts-{VECTORIZER_FEAT}"
assert src_dir.exists(), f"{src_dir} doesn't exist"
out_dir = Path("/home/iailab36/iser/models") / f"corex-sts-{N_HIDDEN}"
out_dir.mkdir(exist_ok=True, parents=True)


print("Loading doc_word.")
doc_word = ss.load_npz(src_dir / "doc_word.npz")
np.random.seed(SEED)
perm = np.random.permutation(doc_word.shape[0])
doc_word = doc_word[perm][:1_500_000]
del perm

print(doc_word.shape)

print("Training topic model.")
# Train the CorEx topic model with 50 topics
topic_model = ct.Corex(n_hidden=N_HIDDEN, max_iter=MAX_ITER, seed=SEED, verbose=True)
topic_model.fit(doc_word)

print("Saving topic model.")
# save topic model
topic_model.save(out_dir / "corex_model.bin")
