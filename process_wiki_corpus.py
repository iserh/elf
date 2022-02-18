from pathlib import Path

import pandas as pd
from gensim.corpora import WikiCorpus
from tqdm import trange
from utils import tokenizer_factory

data_dir = Path("/home/iailab36/iser/data/wiki")
output_dir = Path("/home/iailab36/iser/data/wiki")
output_dir.mkdir(exist_ok=True)

SAVE_INTERVAL = 10_000

wiki = WikiCorpus(
    data_dir / "enwiki-latest-pages-articles.xml.bz2",
    dictionary={},
    processes=20,
    tokenizer_func=tokenizer_factory(),
)


articles = []
with trange(SAVE_INTERVAL, leave=False) as pbar:
    pbar.set_description_str(f"preprocessing")

    for i, tokens in enumerate(wiki.get_texts(), start=1):
        articles.append(" ".join(tokens))

        # update pbar
        pbar.update()

        if i % SAVE_INTERVAL == 0:
            # save batch
            df = pd.DataFrame({"article": articles})
            df.to_feather(output_dir / f"wiki_preprocessed.feather")
            # free memory
            del df, articles
            articles = []
            # increase batch counter
            j += 1
            # reset pbar
            pbar.reset(total=i + SAVE_INTERVAL)
            pbar.update(i)

# save last batch
if len(articles) > 0:
    # save batch
    df = pd.DataFrame({"article": articles})
    df.to_feather(output_dir / f"wiki_batch_{j:02d}.feather")

print(f"finished: saved {i} articles.")

# df = pd.read_feather(output_dir / f"wiki_batch_01.feather")
# print(df.article[0])
