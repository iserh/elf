from pathlib import Path

import pandas as pd
from gensim.corpora import WikiCorpus
from tqdm import trange

from utils import tokenize

data_dir = Path("/home/iailab36/iser/data/wiki")
output_dir = Path("/home/iailab36/iser/data/wiki")
output_dir.mkdir(exist_ok=True)

SAVE_INTERVAL = 100_000
UPDATE_INTERVAL = 100

wiki = WikiCorpus(
    data_dir / "enwiki-latest-pages-articles.xml.bz2",
    dictionary={},
    processes=20,
    tokenizer_func=tokenize,
)

docs = []

# process wiki corpus
with trange(SAVE_INTERVAL, leave=False, desc="preprocessing") as pbar:
    for i, tokens in enumerate(wiki.get_texts(), start=1):
        # append doc to list
        docs.append(" ".join(tokens))

        if i % UPDATE_INTERVAL == 0:
            # update pbar
            pbar.update(n=UPDATE_INTERVAL)

        if i % SAVE_INTERVAL == 0:
            # save batch
            df = pd.DataFrame({"doc": docs})
            df.to_feather(output_dir / "wiki_preprocessed.feather")
            # free memory
            del df
            # reset pbar
            pbar.reset(total=i + SAVE_INTERVAL)
            pbar.update(i)

# final save
df = pd.DataFrame({"doc": docs})
df.to_feather(output_dir / "wiki_preprocessed.feather")

print(f"finished: saved {i} articles.")
