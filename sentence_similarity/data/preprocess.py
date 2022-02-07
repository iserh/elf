import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import pandas as pd
import spacy
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
    filtered_pos_tags: List[str] = field(default_factory=lambda: ["X", "PUNCT", "SYM"])
    use_lemmas: bool = True
    remove_stop_words: bool = True
    remove_numbers: bool = True
    remove_symbols: bool = True
    remove_punctuation: bool = True

    def save(self, fpath: Path) -> None:
        with open(fpath, "w") as f:
            json.dump(self.__dict__, f)
    
    @classmethod
    def load(cls, fpath: Path) -> "PipelineConfig":
        with open(fpath, "r") as f:
            config = cls(json.load(f))
        return config


class Pipeline:
    def __init__(self, config: PipelineConfig = PipelineConfig()) -> None:
        self.config = config
        # spacy language pipeline
        self.nlp = spacy.load("en_core_web_lg", exclude=["parser", "ner"])
        # configuration
        self.filtered_pos_tags = config.filtered_pos_tags
        self.use_lemmas = config.use_lemmas
        self.remove_stop_words = config.remove_stop_words
        # regular expressions for cleaning
        clean_regex = []
        if config.remove_numbers:
            clean_regex.append("[0-9]+[\,,.]?[0-9]+")
        if config.remove_symbols:
            clean_regex.append("[\$#]+")
        if config.remove_punctuation:
            clean_regex.append("[\,.\?]+")
        # compile regex
        self.cleanr = re.compile("|".join(clean_regex))

    def __call__(self, sentences: pd.Series, split_tokens=False) -> pd.Series:
        """Only keep lemmatized version of words with allowed POS tags."""
        # spacy processing
        processed = []
        for doc in tqdm(self.nlp.pipe(sentences), total=len(sentences), desc="Preprocessing"):
            processed.append("".join([
                f"{token.lemma_ if self.use_lemmas else token.text} "
                for token in doc if (
                    not token.pos_ in self.filtered_pos_tags and
                    not (self.remove_stop_words and token.is_stop)
                )
            ]))
        # cast to pandas series again
        processed = pd.Series(processed, index=sentences.index)
        # regex processing
        processed = processed.apply(lambda s: re.sub(self.cleanr, '', s))
        if split_tokens:
            processed = processed.apply(lambda s: [t for t in s.split(' ') if t != ''])
        return processed
