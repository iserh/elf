import pickle
from pathlib import Path

import corextopic.corextopic as ct
import gensim
import numpy as np
import spacy

from .preprocess import Pipeline, PipelineConfig


class CoreXFeatures:
    def __init__(self, data_dir: Path):
        super().__init__()
        assert data_dir.exists()
        # initialize preprocessing pipeline
        config = PipelineConfig.load(data_dir / "pipe_config.json")
        self.pipeline = Pipeline(config)
        # load vectorizer
        with open(data_dir / "vectorizer.bin", "rb") as f:
            self.vectorizer = pickle.load(f)
        # load topic model
        self.corex: ct.Corex = ct.load(data_dir / "corex_model.bin")
    
    @property
    def input_size(self):
        return self.corex.n_hidden

    def __call__(self, s):
        # preprocess
        s = self.pipeline(s)
        # vectorize
        s = self.vectorizer.transform(s)
        # apply topic model
        probs = self.corex.transform(s, details=True)[0]
        # convert to tensor
        return np.array(probs, dtype=np.float32)


class LDAFeatures:
    def __init__(self, data_dir: Path):
        super().__init__()
        assert data_dir.exists()
        # initialize preprocessing pipeline
        config = PipelineConfig.load(data_dir)
        self.pipeline = Pipeline(config)
        # load lda
        self.lda = gensim.models.LdaMulticore.load(str(data_dir / "lda"))
    
    @property
    def input_size(self):
        return self.lda.num_topics

    def __call__(self, s):
        # preprocess
        s = self.pipeline(s, split_tokens=True)
        # vectorize
        s = [self.lda.id2word.doc2bow(tokens) for tokens in s]
        # apply topic model
        probs = probs = np.array([[topic[1] for topic in out] for out in self.lda[s]])
        # convert to tensor
        return np.array(probs, dtype=np.float32)


class SyntaxFeatures:
    def __init__(self):
        super().__init__()
        # spacy language pipeline
        self.nlp = spacy.load("en_core_web_lg", exclude=["ner"])
    
    @property
    def input_size(self):
        return 3

    def __call__(self, sentences):
        # spacy processing
        processed = []
        for doc in self.nlp.pipe(sentences):
            nsubj = 0
            dobj = 0
            for token in doc:
                if token.dep_ == "ROOT":
                    root = token.pos
                elif token.dep_ == "nsubj":
                    nsubj = token.pos
                elif token.dep_ == "nobj":
                    dobj = token.pos
            processed.append(np.stack([root, nsubj, dobj]))
        return np.array(np.stack(processed), dtype=np.int32)
