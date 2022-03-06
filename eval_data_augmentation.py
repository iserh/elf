from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils import (CoreXProbsFactory, LDAProbs, SyntaxFactory, preprocess,
                   tokenize)

data_dir = Path(".")
model_dir = Path("/home/iailab36/iser/models")

SEEDS = [1337, 89, 56, 23, 54]
COREX_HIDDEN_LIST = [16, 32, 64, 128, 300]
VEC_FEAT = 10_000
DATA_AUGMENTATION = False
SYNTAX = True


random_forest = lambda SEED: RandomForestRegressor(criterion="squared_error", n_estimators=100, random_state=SEED)
decision_tree = lambda SEED: DecisionTreeRegressor(random_state=SEED, max_depth=5)
mlp = lambda SEED: MLPRegressor((512, 256), random_state=SEED)


# Load benchmark dataset
train_data = pd.read_feather(data_dir / "stsbenchmark" / f"sts-train{'-sbert' if DATA_AUGMENTATION else ''}.feather")
val_data = pd.read_feather(data_dir / "stsbenchmark" / f"sts-dev{'-sbert' if DATA_AUGMENTATION else ''}.feather")
test_data = pd.read_feather(data_dir / "stsbenchmark" / "sts-test.feather")

train_data = pd.concat([train_data, val_data]).reset_index(drop=True)


COLUMNS = ["N_HIDDEN", "random_forest", "decision_tree", "mlp"]
df_results = pd.DataFrame(columns=COLUMNS, dtype=float)
df_results_aug = pd.DataFrame(columns=COLUMNS, dtype=float)


for COREX_HIDDEN in COREX_HIDDEN_LIST:
    print(f"{COREX_HIDDEN=}")
    results = [COREX_HIDDEN]

    features_train = []
    features_val = []


    get_topic_probs = CoreXProbsFactory(
        vectorizer_path=model_dir / f"sts_vec={VEC_FEAT}",
        corex_name=f"corex_n_hidden={COREX_HIDDEN}_iter=7",
    )


    # compute topic probabilities
    topic_probs_train_1 = get_topic_probs(train_data.s1)
    topic_probs_train_2 = get_topic_probs(train_data.s2)
    topic_probs_test_1 = get_topic_probs(test_data.s1)
    topic_probs_test_2 = get_topic_probs(test_data.s2)
    # concatenate topics of the two sentences
    topic_probs_train = np.concatenate([topic_probs_train_1, topic_probs_train_2], axis=1)
    topic_probs_test = np.concatenate([topic_probs_test_1, topic_probs_test_2], axis=1)
    # add to features list
    features_train.append(topic_probs_train)
    features_val.append(topic_probs_test)


    if SYNTAX:
        get_syntax_deps = SyntaxFactory()
        # compute syntax tokens
        syntax_train_1 = get_syntax_deps(train_data.s1)
        syntax_train_2 = get_syntax_deps(train_data.s2)
        syntax_test_1 = get_syntax_deps(test_data.s1)
        syntax_test_2 = get_syntax_deps(test_data.s2)
        # mask matching syntax
        syntax_train = (syntax_train_1 == syntax_train_2).astype(int)
        syntax_test = (syntax_test_1 == syntax_test_2).astype(int)
        # append to features list
        features_train.append(syntax_train)
        features_val.append(syntax_test)


    # create input vectors
    X_train = np.concatenate(features_train, axis=1)
    X_test = np.concatenate(features_val, axis=1)
    # create targets
    y_train = train_data.score
    y_test = test_data.score
    print("X_train:", X_train.shape)


    if DATA_AUGMENTATION:
        # load augmentation dataset
        aug_data = pd.read_feather(data_dir / "stsbenchmark" / "df_augment.feather")

        features_aug = []


        # get topics of the augmented sentences
        topic_probs_augmented = np.concatenate([
            topic_probs_train_1[aug_data.idx1],
            topic_probs_train_2[aug_data.idx2]
        ], axis=1)
        features_aug.append(topic_probs_augmented)


        if SYNTAX:
            # syntax features
            syntax_aug = (syntax_train_1[aug_data.idx1] == syntax_train_2[aug_data.idx2]).astype(int)
            features_aug.append(syntax_aug)


        # create inputs / targets of augmented dataset
        X_aug = np.concatenate(features_aug, axis=1)
        y_aug = aug_data.score
        print(f"#augmented: {y_aug.shape[0]}")

        X_train_w_aug = np.concatenate([X_train, X_aug])
        y_train_w_aug = np.concatenate([y_train, y_aug])
        print(f"#(train+augmented): {y_train_w_aug.shape[0]}")

        X_train = X_train_w_aug
        y_train = y_train_w_aug


    for model_cls in [random_forest, decision_tree, mlp]:
        metrics = np.empty((len(SEEDS),))

        for i, SEED in enumerate((pbar := tqdm(SEEDS, desc=model_cls(0).__class__.__name__, leave=False))):
            np.random.seed(SEED)
            perm = np.random.permutation(X_train.shape[0])
            X_train_ = X_train[perm]
            y_train_ = y_train[perm]

            model = model_cls(SEED)
            ignore_warnings(category=ConvergenceWarning)(
                model.fit(X_train_, y_train_)
            )

            # evaluate model
            spearman_train = spearmanr(model.predict(X_train), y_train)[0]
            spearman_test = spearmanr(model.predict(X_test), y_test)[0]
            metrics[i] = spearman_test
            pbar.set_postfix({"p": metrics[i]})

        results.append(metrics.mean())
        print(f"SpearmanRank: {results[-1]:.4f}")

    df_results = pd.concat([
        df_results,
        pd.DataFrame([results], columns=COLUMNS),
    ])
    df_results.T.to_csv(f"./df_results{'_aug' if DATA_AUGMENTATION else ''}{'_syn' if SYNTAX else ''}.csv")
