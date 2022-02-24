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
    def __init__(self, corex_path: Path, vectorizer_path: Path) -> None:
        with open(vectorizer_path / "vectorizer.bin", "rb") as f:
            self.vectorizer: TfidfVectorizer = pickle.load(f)
        self.corex: ct.Corex = ct.load(corex_path / "corex_model.bin")

    def __call__(self, docs):
        processed = map(preprocess, docs)
        X = self.vectorizer.transform(processed)
        topic_probs = self.corex.transform(X, details=True)[0]
        return topic_probs


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
