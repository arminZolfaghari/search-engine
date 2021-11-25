import pandas as pd
from hazm import *
# from __future__ import unicode_literals

FILE_NAME = "./IR1_7k_news.xlsx"
normalizer = Normalizer()


def read_data_from_file(file_name):
    data_frame = pd.read_excel(file_name, sheet_name='Sheet1', usecols="A,C")
    # print(data_frame)
    return data_frame


def normalize(sentence):
    normalized_sentence = normalizer.normalize(sentence)
    return normalized_sentence


def preprocess(data_frame, ):
    new_data_frame = data_frame

    # step1: normalize sentences
    new_data_frame['A'] = data_frame['A'].apply(normalize)

    # step2: Tokenization


if __name__ == "__main__":
    df = read_data_from_file(FILE_NAME)
    preprocess(df)
