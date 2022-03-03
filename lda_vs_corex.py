from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils._testing import ignore_warnings
from tqdm import tqdm

from utils import CoreXProbsFactory, LDAProbs

data_dir = Path(".")
model_dir = Path("/home/iailab36/iser/models")

SEEDS = [1337, 89, 56, 23, 54]
N_HIDDEN_LIST = [16, 32, 64, 128, 300]
VEC_FEAT = 10_000


# Load benchmark dataset
train_data = pd.read_feather(data_dir / "stsbenchmark" / "sts-train.feather")
val_data = pd.read_feather(data_dir / "stsbenchmark" / "sts-dev.feather")
test_data = pd.read_feather(data_dir / "stsbenchmark" / "sts-test.feather")

train_data = pd.concat([train_data, val_data]).reset_index(drop=True)


random_forest = lambda SEED: RandomForestRegressor(criterion="squared_error", n_estimators=100, random_state=SEED)
decision_tree = lambda SEED: DecisionTreeRegressor(random_state=SEED, max_depth=5)
mlp = lambda SEED: MLPRegressor((512, 256), random_state=SEED)


for N_HIDDEN in N_HIDDEN_LIST:
    print()
    print()
    print(f"train configuration: N_HIDDEN={N_HIDDEN}")
    print()

    for get_topic_probs in [
        CoreXProbsFactory(
            vectorizer_path=model_dir / f"sts_vec={VEC_FEAT}",
            corex_name=f"corex_n_hidden={N_HIDDEN}_iter=7",
        ),
        LDAProbs(model_dir / f"sts_lda_hidden={N_HIDDEN}"),
    ]:
        print(get_topic_probs.__class__.__name__)
        print()

        # compute topic probabilities
        topic_probs_train_1 = get_topic_probs(train_data.s1)
        topic_probs_train_2 = get_topic_probs(train_data.s2)
        topic_probs_test_1 = get_topic_probs(test_data.s1)
        topic_probs_test_2 = get_topic_probs(test_data.s2)

        # concatenate topics of the two sentences
        X_train = np.concatenate([topic_probs_train_1, topic_probs_train_2], axis=1)
        X_test = np.concatenate([topic_probs_test_1, topic_probs_test_2], axis=1)

        # create targets
        y_train = train_data.score
        y_val = test_data.score
        print("X_train:", X_train.shape)


        for model_type in [random_forest]:
            print(model_type(0).__class__.__name__)

            spearman_train = np.empty((len(SEEDS,)))
            spearman_test = np.empty((len(SEEDS,)))

            for i, SEED in enumerate(tqdm(SEEDS, leave=False)):
                np.random.seed(SEED)
                perm = np.random.permutation(X_train.shape[0])
                X_train_ = X_train[perm]
                y_train_ = y_train[perm]

                model = model_type(SEED)
                ignore_warnings(category=ConvergenceWarning)(
                    model.fit(X_train_, y_train_)
                )

                # evaluate model
                spearman_train[i] = spearmanr(model.predict(X_train), y_train)[0]
                spearman_test[i] = spearmanr(model.predict(X_test), y_val)[0]

                # print(f"SpearmanRank-train: {spearman_train[i]:.4f},\t SpearmanRank-test: {spearman_test[i]:.4f}")

            print(f"Mean & Std for {model.__class__.__name__}")
            print(f"SpearmanRank-train: mean={spearman_train.mean():.4f}, std={spearman_train.std():.4f}")
            print(f"SpearmanRank-test: mean={spearman_test.mean():.4f}, std={spearman_test.std():.4f}")
            print()
