import json

import pandas as pd
from hazm import *
from DictList import *
# from __future__ import unicode_literals
import DictList

FILE_NAME = "./IR1_7k_news.xlsx"
normalizer = Normalizer()
stemmer = Stemmer()


def save_data_frame_to_file(data_frame):
    data_frame.to_excel("training_data.xlsx")


def read_data_from_file(file_name=FILE_NAME):
    data_frame = pd.read_excel(file_name, sheet_name='Sheet1', usecols="A,C,D")
    return data_frame


def normalize(sentence):
    normalized_sentence = normalizer.normalize(sentence)
    return normalized_sentence


# get sentence then return a list of tokens with indexes
def tokenize(sentence, type):
    tokens = word_tokenize(sentence)

    if type == "positional":
        # create dictionary tokens with indexes in sentences
        tokens_index_dict = DictList.DictList()
        for index in range(len(tokens)):
            tokens_index_dict[tokens[index]] = index
        return tokens_index_dict

    else:  # non positional
        tokens_list = tokens
        return tokens_list


def stem(words_dict, type):
    """
    get list of words and return list of stem words
    :param type:
    :param words_dict: dictionary of words with index
    :return new_words_dict: return dictionary of words stem with index
    """

    if type == "positional":
        new_words_dict = DictList.DictList()
        for key, value in words_dict.items():
            word_stem = stemmer.stem(key)
            for i in range(len(value)):
                new_words_dict[word_stem] = value[i]

        return new_words_dict

    else:  # type = non positional
        new_tokens_list = []
        for token in words_dict:
            stem_token = stemmer.stem(token)
            new_tokens_list.append(stem_token)

        return new_tokens_list


def remove_stop_words(words_dict, type):
    """
    :param type:
    :param words_dict: dictionary of words with index
    :return new_words_dict: return dictionary of words with index and without stop words
    """

    stop_words = list(set(stopwords_list()).intersection(list(words_dict)))
    if type == "positional":
        # new_words_dict = copy.deepcopy(words_dict)
        for word in stop_words:
            del words_dict[word]

    else:  # type = non positional
        for word in stop_words:
            words_dict.remove(word)

    return words_dict


def preprocess(data_frame, type, remove_stop_words_flag=False, stem_flag=False):
    new_data_frame = data_frame

    # step1: Normalize
    new_data_frame['content'] = data_frame['content'].apply(normalize)

    # step2: Tokenization
    new_data_frame['tokens'] = new_data_frame['content'].apply(tokenize, type=type)
    # print(new_data_frame["tokens"][5])

    # step3: Stemming
    if stem_flag:
        new_data_frame['tokens'] = new_data_frame['tokens'].apply(stem, type=type)
    # print(new_data_frame["tokens"][5])

    # print(len(new_data_frame["tokens"][0]))
    # step4: Stop words
    if remove_stop_words_flag:
        new_data_frame['tokens'] = new_data_frame['tokens'].apply(remove_stop_words, type=type)
        # print(len(new_data_frame["tokens"][0]))

    return new_data_frame


def preprocess_query(query, type, remove_stop_words_flag=False, stem_flag=False):
    # step1: Normalize
    normalize_query = normalize(query)

    # step2: Tokenization
    query_tokens_dict = tokenize(normalize_query, type)

    # step3: Stemming
    if stem_flag:
        query_tokens_dict = stem(query_tokens_dict, type)

    # step4: Stop words
    if remove_stop_words_flag:
        query_tokens_dict = remove_stop_words(query_tokens_dict, type)

    return query_tokens_dict


def preprocess_word(word):
    # step1: Normalize
    new_word = normalizer.normalize(word)

    # step3: Stemming
    new_word = stemmer.stem(new_word)

    return new_word


def get_data_frame_after_preprocess(type, remove_stop_words_flag, stem_flag):
    data_frame = read_data_from_file(FILE_NAME)
    return preprocess(data_frame, type, remove_stop_words_flag, stem_flag)


if __name__ == "__main__":
    print()
    df = read_data_from_file()
    data_frame_after_preprocess = preprocess(df, "non positional", True, True)
    # save_data_frame_to_file(data_frame_after_preprocess)

    # training_data_list = []
    # for doc_id, row in data_frame_after_preprocess.iterrows():
    #     training_data_list.append(row["tokens"])
    #
    # with open("training_data.json", 'w', encoding='utf-8') as fp:
    #     json.dump(training_data_list, fp, sort_keys=True, indent=4, ensure_ascii=False)
