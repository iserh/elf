import corextopic.corextopic as ct
from typing import Callable
from pathlib import Path

from gensim.utils import to_unicode, deaccent
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import re
from pathlib import Path
import numpy as np
import spacy

import pickle


def tokenizer_factory() -> Callable:
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

    return tokenize


def preprocess_factory():
    tokenize = tokenizer_factory()

    def preprocess(doc):
        return " ".join(tokenize(doc))

    return preprocess


def corex_probs_factory(corex_path: Path, vectorizer_path: Path):

    preprocess = preprocess_factory()
    
    with open(vectorizer_path / "vectorizer.bin", "rb") as f:
        vectorizer: TfidfVectorizer = pickle.load(f)
    corex: ct.Corex = ct.load(corex_path / "corex_model.bin")

    def get_topic_probs(docs):
        processed = map(preprocess, docs)
        X = vectorizer.transform(processed)
        topic_probs = corex.transform(X, details=True)[0]

        return topic_probs

    return get_topic_probs

def syntax_factory():
    nlp = spacy.load("en_core_web_lg", exclude=["ner"])

    def get_syntax_deps(docs):
        deps = []
        for doc in nlp.pipe(docs):
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

    return get_syntax_deps
