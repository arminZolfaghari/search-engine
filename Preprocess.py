import pandas as pd
from hazm import *
from DictList import *
# from __future__ import unicode_literals
import DictList

FILE_NAME = "./IR1_7k_news.xlsx"
normalizer = Normalizer()
stemmer = Stemmer()


def save_data_frame_to_file(data_frame):
    data_frame.to_excel("df_after_preprocess_without_stop_words.xlsx")


def read_data_from_file(file_name=FILE_NAME):
    data_frame = pd.read_excel(file_name, sheet_name='Sheet1', usecols="A,C")
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
    new_words_dict = DictList.DictList()
    for key, value in words_dict.items():
        word_stem = stemmer.stem(key)
        for i in range(len(value)):
            new_words_dict[word_stem] = value[i]

    return new_words_dict


def remove_stop_words(words_dict):
    """
    :param words_dict: dictionary of words with index
    :return new_words_dict: return dictionary of words with index and without stop words
    """

    stop_words = list(set(stopwords_list()).intersection(list(words_dict)))
    # new_words_dict = copy.deepcopy(words_dict)
    for word in stop_words:
        del words_dict[word]

    return words_dict


def preprocess(data_frame, remove_stop_words_flag=False, stem_flag=False):
    new_data_frame = data_frame

    # step1: Normalize
    new_data_frame['content'] = data_frame['content'].apply(normalize)

    # step2: Tokenization
    new_data_frame['tokens'] = new_data_frame['content'].apply(tokenize)
    # print(new_data_frame["tokens"][5])

    # step3: Stemming
    if stem_flag:
        new_data_frame['tokens'] = new_data_frame['tokens'].apply(stem)
    # print(new_data_frame["tokens"][5])

    # print(len(new_data_frame["tokens"][0]))
    # step4: Stop words
    if remove_stop_words_flag:
        new_data_frame['tokens'] = new_data_frame['tokens'].apply(remove_stop_words)
        # print(len(new_data_frame["tokens"][0]))

    return new_data_frame


def preprocess_word(word):
    # step1: Normalize
    new_word = normalizer.normalize(word)

    # step3: Stemming
    new_word = stemmer.stem(new_word)

    return new_word


def get_data_frame_after_preprocess(remove_stop_words_flag, stem_flag):
    data_frame = read_data_from_file(FILE_NAME)
    return preprocess(data_frame, remove_stop_words_flag, stem_flag)


if __name__ == "__main__":
    print()
    df = read_data_from_file()
    data_frame_after_preprocess = preprocess(df)
    save_data_frame_to_file(data_frame_after_preprocess)
