"""Training the topic model."""
from pathlib import Path

import gensim
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import tokenize

N_HIDDEN = 300
SEED = 1337
np.random.seed(SEED)

src_dir = Path("/home/iailab36/iser/data/stsbenchmark")
assert src_dir.exists()
out_dir = Path("/home/iailab36/iser/models") / f"sts_lda_hidden={N_HIDDEN}"
out_dir.mkdir(exist_ok=True, parents=True)

df_train = pd.read_feather(src_dir / "sts-train.feather")
df_val = pd.read_feather(src_dir / "sts-dev.feather")
df = pd.concat([df_train, df_val])
docs = pd.concat([df.s1, df.s2])
# docs = df.doc

del df_train, df_val, df

# preprocessing
tqdm.pandas()
docs = docs.progress_apply(tokenize)

id2word = gensim.corpora.Dictionary(docs)

# create lda model
lda = gensim.models.LdaMulticore(
   corpus=[id2word.doc2bow(doc) for doc in docs],
   num_topics=N_HIDDEN,
   id2word=id2word,
   workers=10,
   random_state=SEED,
   minimum_probability=0,
)
# save
lda.save(str(out_dir / "lda"))
