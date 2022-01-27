## TODO
- syntax dependencies? (different implementation with 0/1)
- error analysis for better features
- data augmentation
- does topic model need more (outside) data?
    - Hypothesis: Pre-Trained models can generalize, because they have seen the whole language. -> Topic modeling can, by definition, not yield good results, because it can only model the train topics (would require huge dataset).
- Other dataset

## DONE
- Only use Nouns & Verbs
- Lemmatization
- ngrams



## Summary

- Lemmatization, preprocessing improvements
- ngrams didn't work
- data augmentation difficult, because no new data is seen
    - data augmentation method needs additional unlabelled data which isn't there for the provided datasets
- syntax does improve slightly
- 

- first experiments on QQP show no improvement on that dataset (67,8% accuracy on train)







## --------------------------------------------------------------

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
