from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import re

import pandas as pd
from tqdm import tqdm


stemmer = SnowballStemmer("english")
stop_words = set(stopwords.words('english'))

filter_sym = re.compile(r'[^a-zA-Z\d\s\.,:]', re.UNICODE)
filter_num = re.compile(r"[0-9]+([,\.]?[0-9])*", re.UNICODE)


def process_doc(doc: str):
    return " ".join([stemmer.stem(token) for token in doc.split(" ") if not token in stop_words])

df: pd.DataFrame = pd.read_feather("/home/iailab36/iser/data/wiki/wiki_processed.feather")
tqdm.pandas(desc="preprocess")
df.doc = df.doc.progress_apply(process_doc)

df.to_feather("/home/iailab36/iser/data/wiki/wiki_processed_.feather")
