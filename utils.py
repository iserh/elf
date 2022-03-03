import pickle
import re
from pathlib import Path

import corextopic.corextopic as ct
import numpy as np
import spacy
from gensim.utils import deaccent, to_unicode
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import gensim

stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words('english'))

filter_sym = re.compile(r'[^a-zA-Z\d\s\.,:]', re.UNICODE)
filter_num = re.compile(r"[0-9]+([,\.]?[0-9])*", re.UNICODE)


def tokenize(content, token_min_len=2, token_max_len=15, lower=True):
    content = to_unicode(content)

    content = content.lower()
    content = deaccent(content)

    content = filter_sym.sub("", content)
    content = filter_num.sub(" num ", content)

    return [
        stemmer.stem(token) for token in word_tokenize(content)
        if not token in stop_words
    ]


def preprocess(doc):
    return " ".join(tokenize(doc))


class CoreXProbsFactory:
    def __init__(self, vectorizer_path: Path, corex_name: str) -> None:
        with open(vectorizer_path / "vectorizer.bin", "rb") as f:
            self.vectorizer: TfidfVectorizer = pickle.load(f)
        self.corex: ct.Corex = ct.load(vectorizer_path / corex_name / "corex_model.bin")

    def __call__(self, docs):
        processed = map(preprocess, docs)
        X = self.vectorizer.transform(processed)
        topic_probs = self.corex.transform(X, details=True)[0]
        return topic_probs


class LDAProbs:
    def __init__(self, path: Path) -> None:
        super().__init__()
        assert path.exists()
        # load lda
        self.lda = gensim.models.LdaMulticore.load(str(path / "lda"))

    def __call__(self, docs):
        docs = map(tokenize, docs)
        # vectorize
        bow = [self.lda.id2word.doc2bow(tokens) for tokens in docs]
        # apply topic model
        probs = [[topic[1] for topic in out] for out in self.lda[bow]]
        # return as np array
        return np.array(probs)


class SyntaxFactory:
    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_lg", exclude=["ner"])

    def __call__(self, docs):
        deps = []
        for doc in self.nlp.pipe(docs):
            nsubj = 0
            dobj = 0
            for token in doc:
                if token.dep_ == "ROOT":
                    root = token.lemma_
                elif token.dep_ == "nsubj":
                    nsubj = token.lemma_
                elif token.dep_ == "nobj":
                    dobj = token.lemma_
            deps.append(np.stack([root, nsubj, dobj]))
        return np.stack(deps)


def prepare_stsbenchmark(path: Path):
    for partition in ["train", "dev", "test"]:
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
