import pandas as pd
from hazm import *
from DictList import *
import copy
# from __future__ import unicode_literals
import DictList

FILE_NAME = "./IR1_7k_news.xlsx"
normalizer = Normalizer()
stemmer = Stemmer()


def save_data_frame_to_file(data_frame):
    print()


def read_data_from_file(file_name):
    data_frame = pd.read_excel(file_name, sheet_name='Sheet1', usecols="A,C")
    # print(data_frame)
    return data_frame


def normalize(sentence):
    normalized_sentence = normalizer.normalize(sentence)
    return normalized_sentence


# get sentence then return a list of tokens with indexes
def tokenize(sentence):
    tokens = word_tokenize(sentence)
    # create dictionary tokens with indexes in sentences
    tokens_index_dict = DictList.DictList()
    for index in range(len(tokens)):
        tokens_index_dict[tokens[index]] = index

    return tokens_index_dict


def stem(words_dict):
    """
    get list of words and return list of stem words
    :param words_dict: dictionary of words with index
    :return new_words_dict: return dictionary of words stem with index
    """
    new_words_dict = copy.deepcopy(words_dict)
    for key, value in words_dict.items():
        word_stem = stemmer.stem(key)
        if word_stem == key:
            continue

        del new_words_dict[key]
        new_words_dict[word_stem] = value[0]

    return new_words_dict


def remove_stop_words(words_dict):
    """
    :param words_dict: dictionary of words with index
    :return new_words_dict: return dictionary of words with index and without stop words
    """

    new_words_dict = copy.deepcopy(words_dict)
    for key in words_dict:
        if key in stopwords_list():
            del new_words_dict[key]

    return new_words_dict


def preprocess(data_frame, remove_stop_words_flag=False):
    new_data_frame = data_frame

    # step1: Normalize
    new_data_frame['content'] = data_frame['content'].apply(normalize)

    # step2: Tokenization
    new_data_frame['tokens'] = new_data_frame['content'].apply(tokenize)
    # print(new_data_frame["tokens"][0]['مسلم'])

    # step3: Stemming
    new_data_frame['tokens'] = new_data_frame['tokens'].apply(stem)
    print(len(new_data_frame["tokens"][0]))

    # step4: Stop words
    if remove_stop_words_flag:
        new_data_frame['tokens'] = new_data_frame['tokens'].apply(remove_stop_words)
        print(len(new_data_frame["tokens"][0]))


if __name__ == "__main__":
    df = read_data_from_file(FILE_NAME)
    preprocess(df)
