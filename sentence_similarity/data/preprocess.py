import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd
import spacy
from spacy.lang.en import English
from tqdm import tqdm


@dataclass
class PipelineConfig:
    """Configuration of Pipeline.
    
    Args:
        `filtered_pos_tags`: Part-Of-Speach tags to remove
        `use_lemmas`: Replace words with lemmatization
        `remove_numbers`: remove numbers
        `remove_symbols`: remove symbols like #,$
        `remove_punctuation`: remove commas, puncts and question marks
    """
    use_lemmas: bool = True
    remove_stop_words: bool = True
    remove_numbers: bool = False
    remove_symbols: bool = False
    remove_punctuation: bool = False

    def save(self, fpath: Path) -> None:
        with open(fpath, "w") as f:
            json.dump(self.__dict__, f)
    
    @classmethod
    def load(cls, fpath: Path) -> "PipelineConfig":
        with open(fpath, "r") as f:
            config = cls(**json.load(f))
        return config


class Pipeline:
    def __init__(self, config: PipelineConfig = PipelineConfig()) -> None:
        self.config = config
        # spacy language pipeline
        self.nlp = spacy.load("en_core_web_lg", exclude=["parser", "ner"])
        # configuration
        self.use_lemmas = config.use_lemmas
        self.remove_stop_words = config.remove_stop_words
        # regular expressions for cleaning
        self.remove_numbers = config.remove_numbers
        self.remove_symbols = config.remove_symbols
        self.remove_punctuation = config.remove_punctuation
        # compile regex
        self.num_regex = re.compile(r"[0-9]+[\,,.]?[0-9]+")
        self.sym_regex = re.compile(r"[\$#]+")
        self.punct_regex = re.compile(r"[\,.\?]+")

    def __call__(self, sentences: pd.Series) -> pd.Series:
        """Only keep lemmatized version of words with allowed POS tags."""
        # spacy processing
        processed = []
        for doc in tqdm(self.nlp.pipe(sentences), total=len(sentences), desc="Preprocessing"):
            processed.append(" ".join([
                token.lemma_ if self.use_lemmas else token.text
                for token in doc if not (self.remove_stop_words and token.is_stop)
            ]))
        # cast to pandas series again
        processed = pd.Series(processed, index=sentences.index)
        # regex processing
        if self.remove_numbers:
            processed = processed.apply(lambda s: re.sub(self.num_regex, 'NUM', s))
        if self.remove_symbols:
            processed = processed.apply(lambda s: re.sub(self.sym_regex, '', s))
        if self.remove_punctuation:
            processed = processed.apply(lambda s: re.sub(self.punct_regex, ' ', s))
        # remove unnecessary whitespaces
        processed = processed.apply(lambda s: " ".join([t for t in s.split(' ') if t != '']))

        return processed
