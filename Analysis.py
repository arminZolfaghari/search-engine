import copy
import math
from Query import *
import matplotlib.pyplot as plt
from Preprocess import *
from PositionalPosting import create_positional_postings_lists

POSITIONAL_POSTINGS_LIST_FILE_WITH_STOP_WORDS = "./positional_postings_lists_with_stop_words.json"
POSITIONAL_POSTINGS_LIST_FILE_WITHOUT_STOP_WORDS = "./positional_postings_lists_without_stop_words.json"


def get_word_freq_dict_from_postings_lists(postings_lists):
    word_freq_dict = {}
    for word, inf_about_word in postings_lists.items():
        word_freq_dict[word] = inf_about_word["frequency_in_all_documents"]

    return word_freq_dict


def remove_high_freq(sorted_word_freq_dict, limit):
    new_sorted_word_freq_dict = copy.deepcopy(sorted_word_freq_dict)
    for i in range(limit):
        word = list(new_sorted_word_freq_dict.keys())[0]
        del new_sorted_word_freq_dict[word]

    return new_sorted_word_freq_dict


def plot_zipf():
    positional_postings_list_with_stop_words = load_positional_postings_list(
        POSITIONAL_POSTINGS_LIST_FILE_WITH_STOP_WORDS)
    positional_postings_list_without_stop_words = load_positional_postings_list(
        POSITIONAL_POSTINGS_LIST_FILE_WITHOUT_STOP_WORDS)

    word_freq_dict_with_stop_words = get_word_freq_dict_from_postings_lists(positional_postings_list_with_stop_words)
    word_freq_dict_without_stop_words = get_word_freq_dict_from_postings_lists(
        positional_postings_list_without_stop_words)

    # sorted descending
    sorted_descending_word_freq_dict_with_stop_words = dict(
        sorted(word_freq_dict_with_stop_words.items(), key=lambda item: item[1], reverse=True))
    sorted_descending_word_freq_dict_without_stop_words = dict(
        sorted(word_freq_dict_without_stop_words.items(), key=lambda item: item[1], reverse=True))
    sorted_descending_word_freq_dict_without_high_freq = remove_high_freq(
        sorted_descending_word_freq_dict_with_stop_words, 30)
    # sorted(word_freq_dict_with_stop_words, reverse=True)
    # sorted(word_freq_dict_without_stop_words, reverse=True)

    max_number_with_stop_words = list(sorted_descending_word_freq_dict_with_stop_words.values())[0]
    max_number_without_stop_words = list(sorted_descending_word_freq_dict_without_stop_words.values())[0]
    max_number_without_high_freq = list(sorted_descending_word_freq_dict_without_high_freq.values())[0]
    print("max_number_with_stop_words: ", max_number_with_stop_words)
    print("max_number_without_stop_words: ", max_number_without_stop_words)
    print("max_number_without_high_freq: ", max_number_without_high_freq)

    # when have stop words
    L1, L2, L3 = [], [], []

    for word, freq in sorted_descending_word_freq_dict_with_stop_words.items():
        L3.append(math.log(freq, 10))
        word_index = list(sorted_descending_word_freq_dict_with_stop_words.keys()).index(word)
        L1.append(math.log(word_index + 1, 10))
        L2.append(math.log(max_number_with_stop_words / (word_index + 1), 10))

    plt.plot(L1, L2)
    plt.plot(L1, L3)
    plt.xlabel("Log 10 Rank")
    plt.ylabel("Log 10 cf")
    plt.title("With stop words")
    plt.show()

    # when remove stop words
    L4, L5, L6 = [], [], []
    for word, freq in sorted_descending_word_freq_dict_without_stop_words.items():
        L6.append(math.log(freq, 10))
        word_index = list(sorted_descending_word_freq_dict_without_stop_words.keys()).index(word)
        L4.append(math.log(word_index + 1, 10))
        L5.append(math.log(max_number_without_stop_words / (word_index + 1), 10))

    plt.plot(L4, L5)
    plt.plot(L4, L6)
    plt.xlabel("Log 10 Rank")
    plt.ylabel("Log 10 cf")
    plt.title("Without stop words")
    plt.show()

    # when remove high freq words
    L7, L8, L9 = [], [], []
    for word, freq in sorted_descending_word_freq_dict_without_high_freq.items():
        L9.append(math.log(freq, 10))
        word_index = list(sorted_descending_word_freq_dict_without_high_freq.keys()).index(word)
        L7.append(math.log(word_index + 1, 10))
        L8.append(math.log(max_number_without_high_freq / (word_index + 1), 10))

    plt.plot(L7, L8)
    plt.plot(L7, L9)
    plt.xlabel("Log 10 Rank")
    plt.ylabel("Log 10 cf")
    plt.title("Without high freq words")
    plt.show()


def calculate_tokens_and_words_number(data_frame):
    postings_lists = create_positional_postings_lists(data_frame)
    number_of_tokens = len(postings_lists)
    number_of_words = 0
    for token, inf_token in postings_lists.items():
        number_of_words += inf_token.frequency_in_all_documents

    return number_of_tokens, number_of_words


def plot_heaps():
    df_after_preprocess_with_stemming = get_data_frame_after_preprocess(True, True)
    df_after_preprocess_without_stemming = get_data_frame_after_preprocess(True, False)

    result_with_stemming = {}
    result_without_stemming = {}
    for number_of_documents in [500, 1000, 1500, 2000]:
        number_of_tokens_with_stemming, number_of_words_with_stemming = calculate_tokens_and_words_number(
            df_after_preprocess_with_stemming.head(number_of_documents))
        result_with_stemming[number_of_documents] = [number_of_tokens_with_stemming, number_of_words_with_stemming]
        number_of_tokens_without_stemming, number_of_words_without_stemming = calculate_tokens_and_words_number(
            df_after_preprocess_without_stemming.head(number_of_documents))
        result_without_stemming[number_of_documents] = [number_of_tokens_without_stemming,
                                                        number_of_words_without_stemming]

    print("with stemming")
    print(result_with_stemming)
    print("without stemming")
    print(result_without_stemming)

    number_of_tokens_with_stemming_in_all_documents, number_of_words_with_stemming_in_all_documents = calculate_tokens_and_words_number(
        df_after_preprocess_with_stemming)
    number_of_tokens_without_stemming_in_all_documents, number_of_words_without_stemming_in_all_documents = calculate_tokens_and_words_number(df_after_preprocess_without_stemming)

    print("with stemming")
    print(f'all documents: number of words: {number_of_tokens_with_stemming_in_all_documents}, number of tokens: {number_of_words_with_stemming_in_all_documents}')
    print("without stemming")
    print(
        f'all documents: number of words: {number_of_tokens_without_stemming_in_all_documents}, number of tokens: {number_of_words_without_stemming_in_all_documents}')



if __name__ == "__main__":
    plot_heaps()
    # plot_zipf()
