import csv
from pathlib import Path

import pandas as pd
from gensim.corpora import WikiCorpus


def STSB(path: Path, partition: str) -> pd.DataFrame:
    # load stsbenchmark feather file
    if (path / "stsbenchmark" / f"sts-{partition}.feather").exists():
        df = pd.read_feather(path / "stsbenchmark" / f"sts-{partition}.feather")
    else:
        # if dataframe doesn't exist as feather, load the csv file
        df: pd.DataFrame = pd.read_csv(path / "stsbenchmark" / f"sts-{partition}.csv", error_bad_lines=False, header = None, delimiter="\t", quoting=csv.QUOTE_NONE, encoding="utf-8")
        # rename columns
        df = df.rename(columns={0: "genre", 1: "filename", 2: "year", 3: "trash", 4: "score", 5: "s1", 6: "s2"})
        # set datatypes
        df.genre = df.genre.astype("category")
        df.filename = df.filename.astype("category")
        df.year = df.year.astype("category")
        df.genre = df.genre.astype("category")
        df.score = df.score / 5
        # save feather
        df.to_feather(path / "stsbenchmark" / f"sts-{partition}.feather")

    return df


def QQP(path: Path, partition: str) -> pd.DataFrame:
    df = pd.read_csv(path / "qqp" / f"{partition}.csv")
    df = df.drop(columns=["id", "qid1", "qid2"])
    df = df.dropna()
    df = df.rename(columns={"question1": "s1", "question2": "s2", "is_duplicate": "score"})


def Wikipedia(path: Path) -> pd.DataFrame:
    pass


def extract_wiki_corpus(path: Path):
    wiki = WikiCorpus(
        path / "enwiki-latest-pages-articles.xml.bz2",
        dictionary={},
        processes=20
    )

    df = pd.DataFrame(columns=["article"], dtype=str)

    output_dir = path / "processed"
    output_dir.mkdir(exist_ok=True)

    print("extracting wiki corpus.")

    j = 1
    for i, text in enumerate(wiki.get_texts(), start=1):
        df = df.append({"article": " ".join(text)}, ignore_index=True)
        if i % 10_000 == 0:
            print(f"processed {i} articles.")
        if i % 100_000 == 0:
            # save batch
            df.to_feather(output_dir / f"wiki_batch_{j:02d}.feather")
            j += 1
            # re init df
            df = pd.DataFrame(columns=["article"], dtype=str)
    
    # save last batch
    if len(df) > 0:
        df.to_feather(output_dir / f"wiki_part_{j:02d}.feather")

    print(f"finished: saved {i} articles.")
