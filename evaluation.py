# %% [markdown]
# # Random Forest

# %% [markdown]
# ### Requirements

# %%
import numpy as np
import pandas as pd

from scipy.stats import spearmanr, pearsonr
from pathlib import Path

data_dir = Path("/home/iailab36/iser/data")
model_dir = Path("/home/iailab36/iser/models")

SEEDS = [1337, 42, 87]
COREX_HIDDEN_LIST = [64]
VEC_FEAT = 10_000

# %%
# Load benchmark dataset
train_data = pd.read_feather(data_dir / "stsbenchmark" / "sts-train.feather")
val_data = pd.read_feather(data_dir / "stsbenchmark" / "sts-dev.feather")
test_data = pd.read_feather(data_dir / "stsbenchmark" / "sts-test.feather")

train_data = pd.concat([train_data, val_data]).reset_index(drop=True)

# %% [markdown]
# ## Pre-compute features

for COREX_HIDDEN in COREX_HIDDEN_LIST:

    # %%
    from utils import CoreXProbsFactory, SyntaxFactory

    features_train = []
    features_val = []

    # %% [markdown]
    # ### topic model features

    # %%
    get_topic_probs = CoreXProbsFactory(
        vectorizer_path=model_dir / f"sts_vec={VEC_FEAT}",
        corex_name=f"corex_n_hidden={COREX_HIDDEN}_iter=7",
    )

    # %%
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

    # %% [markdown]
    # ### syntax features

    # %%
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

    # %% [markdown]
    # ## Training without data augmentation

    # %%
    # create input vectors
    X_train = np.concatenate(features_train, axis=1)
    X_test = np.concatenate(features_val, axis=1)
    # create targets
    y_train = train_data.score
    y_val = test_data.score
    print("X_train:", X_train.shape)

    # %%
    print()
    print()
    print(f"train configuration: COREX_HIDDEN={COREX_HIDDEN}, VEC_FEAT={VEC_FEAT}")
    print()

    # %%
    print("without augmentation")
    print()

    # %%
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.tree import DecisionTreeRegressor

    spearman_train = np.empty((len(SEEDS,)))
    spearman_test = np.empty((len(SEEDS,)))

    rnd_forest = lambda seed: RandomForestRegressor(criterion="squared_error", n_estimators=100, random_state=seed)
    dec_tree = lambda seed: DecisionTreeRegressor(random_state=seed)
    mlp = lambda seed: MLPRegressor((512, 256, 128), random_state=seed)

    for model_cls in [rnd_forest, dec_tree, mlp]:
        print(model_cls(0).__class__.__name__)

        for i, seed in enumerate(SEEDS):

            model = model_cls(seed)
            model.fit(X_train, y_train)

            # evaluate model
            spearman_train[i] = spearmanr(model.predict(X_train), y_train)[0]
            spearman_test[i] = spearmanr(model.predict(X_test), y_val)[0]

            print(f"SpearmanRank-train: {spearman_train[i]:.4f},\t SpearmanRank-test: {spearman_test[i]:.4f}")

        print(f"Mean & Std for {model.__class__.__name__}")
        print(f"SpearmanRank-train: mean={spearman_train.mean():.4f}, std={spearman_train.std():.4f}")
        print(f"SpearmanRank-test: mean={spearman_test.mean():.4f}, std={spearman_test.std():.4f}")
        print()

    # %% [markdown]
    # ## Training with data augmentation

    # %%
    print("with augmentation")

    # %%
    # load augmentation dataset
    aug_data = pd.read_feather("df_augment.feather")

    features_aug = []

    # %%
    # get topics of the augmented sentences
    topic_probs_augmented = np.concatenate([
        topic_probs_train_1[aug_data.idx1],
        topic_probs_train_2[aug_data.idx2]
    ], axis=1)
    features_aug.append(topic_probs_augmented)

    # %%
    # syntax features
    syntax_aug = (syntax_train_1[aug_data.idx1] == syntax_train_2[aug_data.idx2]).astype(int)
    features_aug.append(syntax_aug)

    # %%
    # create inputs / targets of augmented dataset
    X_aug = np.concatenate(features_aug, axis=1)
    y_aug = aug_data.score
    print(f"#augmented: {y_aug.shape[0]}")

    X_train_w_aug = np.concatenate([X_train, X_aug])
    y_train_w_aug = np.concatenate([y_train, y_aug])
    print(f"#(train+augmented): {y_aug.shape[0]}")
    print()

    # %%
    spearman_train = np.empty((len(SEEDS,)))
    spearman_test = np.empty((len(SEEDS,)))

    for model_cls in [rnd_forest, dec_tree, mlp]:
        print(model_cls(0).__class__.__name__)

        for i, seed in enumerate(SEEDS):

            model = model_cls(seed)
            model.fit(X_train_w_aug, y_train_w_aug)

            # evaluate model
            spearman_train[i] = spearmanr(model.predict(X_train), y_train)[0]
            spearman_test[i] = spearmanr(model.predict(X_test), y_val)[0]

            print(f"SpearmanRank-train: {spearman_train[i]:.4f},\t SpearmanRank-test: {spearman_test[i]:.4f}")

        print(f"Mean & Std for {model.__class__.__name__}")
        print(f"SpearmanRank-train: mean={spearman_train.mean():.4f}, std={spearman_train.std():.4f}")
        print(f"SpearmanRank-test: mean={spearman_test.mean():.4f}, std={spearman_test.std():.4f}")
        print()
