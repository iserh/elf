import csv
import sys
from pathlib import Path

import pandas as pd
import six
from gensim.corpora import WikiCorpus

from utils import get_logger

logger = get_logger(__name__)


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


def _process_wiki():
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) != 3:
        print("Using: python process_wiki.py enwiki.xxx.xml.bz2 wiki.en.text")
        sys.exit(1)
    inp, outp = sys.argv[1:3]
    space = " "
    i = 0

    output = open(outp, 'w')
    wiki = WikiCorpus(inp, lemmatize=False, dictionary={})
    for text in wiki.get_texts():
        if six.PY3:
            output.write(bytes(' '.join(text), 'utf-8').decode('utf-8') + '\n')
        #   ###another method###
        #    output.write(
        #            space.join(map(lambda x:x.decode("utf-8"), text)) + '\n')
        else:
            output.write(space.join(text) + "\n")
        i = i + 1
        if (i % 10000 == 0):
            logger.info("Saved " + str(i) + " articles")

    output.close()
    logger.info("Finished Saved " + str(i) + " articles")
