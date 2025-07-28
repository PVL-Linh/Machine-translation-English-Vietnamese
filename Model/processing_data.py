import pandas as pd
# import numpy as np
import re,string

data_dir = "data/data_new/"
en_sents = open(data_dir + 'en_sents', "r",encoding="utf-8").read().splitlines()
vi_sents = open(data_dir + 'vi_sents', "r",encoding="utf-8").read().splitlines()
raw_data = {
        "en": [line for line in en_sents[:254000]], # Only take first 170000 lines
        "vi": [line for line in vi_sents[:254000]],
    }
data_f = pd.DataFrame(raw_data, columns=["en", "vi"])


def preprocessing(df):
    df["en"] = df["en"].apply(
        lambda ele: ele.translate(str.maketrans('', '', string.punctuation)))  # Remove punctuation
    df["vi"] = df["vi"].apply(lambda ele: ele.translate(str.maketrans('', '', string.punctuation)))
    df["en"] = df["en"].apply(lambda ele: ele.lower())  # convert text to lowercase
    df["vi"] = df["vi"].apply(lambda ele: ele.lower())
    df["en"] = df["en"].apply(lambda ele: ele.strip())
    df["vi"] = df["vi"].apply(lambda ele: ele.strip())
    df["en"] = df["en"].apply(lambda ele: re.sub("\s+", " ", ele))
    df["vi"] = df["vi"].apply(lambda ele: re.sub("\s+", " ", ele))

    return df


def processing_data ():
    df = preprocessing(data_f)
    return df

print('Dang xu ly du lieu !!!')
