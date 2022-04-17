import corextopic.corextopic as ct
import scipy.sparse as ss
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_extraction.text import TfidfVectorizer


class CoreX(BaseEstimator, RegressorMixin):
    """An example of classifier."""

    def __init__(
        self,
        vectorizer_max_feat: int = 10_000,
        n_hidden: int = 64,
        corex_max_iter: int = 5,
    ):
        self.vectorizer_max_feat = vectorizer_max_feat
        self.n_hidden = n_hidden
        self.corex_max_iter = corex_max_iter

    def fit(self, X, y=None):
        assert type(self.vectorizer_max_feat) == int, "vectorizer_max_feat parameter must be integer"
        assert type(self.n_hidden) == int, "n_topics parameter must be integer"
        assert type(self.corex_max_iter) == int, "topic_model_max_iter parameter must be integer"

        print("vectorizer fit")
        self.vectorizer_ = TfidfVectorizer(
            max_features=self.vectorizer_max_feat, strip_accents="ascii", lowercase=True, binary=True
        )
        X = self.vectorizer_.fit_transform(X)
        X = ss.csr_matrix(X)

        print("corex fit")
        self.corex_ = ct.Corex(n_hidden=self.n_hidden, max_iter=self.corex_max_iter)
        self.corex_.fit(X)

        return self

    def predict(self, X, y=None):
        X = self.vectorizer_.transform(X)
        return self.corex_.transform(X)

    def score(self, X, y=None):
        return self.corex_.tc


from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV

from elf.utils import preprocess_factory

data_dir = Path("/home/iailab36/iser/data/stsbenchmark")

print("loading data")
data_train = pd.read_feather(data_dir / "sts-train.feather")
data_dev = pd.read_feather(data_dir / "sts-dev.feather")
preprocess = preprocess_factory()

print("preprocessing data")
X_train = pd.concat([data_train.s1, data_train.s2]).apply(preprocess)
X_test = pd.concat([data_dev.s1, data_dev.s2]).apply(preprocess)
tuned_params = {
    "vectorizer_max_feat": [10_000],
    "n_hidden": [10, 50, 64, 84, 100, 128, 200, 256, 300],
    "corex_max_iter": range(2, 20, 1),
}

print("running gridsearch\n")
gs = GridSearchCV(CoreX(), tuned_params)
gs.fit(X_train)

print("\nfinished gridsearch")
print(gs.best_params_)
