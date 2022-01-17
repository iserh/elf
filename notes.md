# Roadmap
## 1. Augment Datasets
* Download the individual datasets
* Use Pre-Trained Sentence BERT to augment the dataset
    * Ensure similarity scores have the same value range
    
### Datasets:
* STS benchmark: http://ixa2.si.ehu.eus/stswiki/index.php/STSbenchmark
* Quora question pairs: https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs
* BWS Argument Similarity Corpus: https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2496.2
* Microsoft Research Paraphrase Corpus: https://github.com/wasiahmad/paraphrase_identification

## 2. Simple explainable features
* Implement simple features like topic models, POS tags, ...
* Train model with these features on (augmented) dataset
* Does this work?
* Does the dataset augmentation improve results?

## 3. Deep Dive: More (complex) explainable features
* Read more literature
* Syntactic features, RegEx
* Combination of features


# Literature
## Topic modeling
### Latent Dirichlet Allocation (LDA)
https://towardsdatascience.com/topic-modeling-and-latent-dirichlet-allocation-in-python-9bf156893c24
### Anchored CorEx: Hierarchical Topic Modeling with Minimal Domain Knowledge
https://github.com/gregversteeg/corex_topic

## Syntactic Features
- Are main verb and noun the same?

## Data Augmentation
### Sentence-bert
https://github.com/UKPLab/sentence-transformers


## --------------------------------------------------------------

# Experiments
## LDA
### Embedding Model
### Linear Model

## CoreX
- TF-IDF Vectorization
### Embedding Model
### Linear Model
### Adjusting number of hidden topics
~50 seems fine, more or less yields worse performance
