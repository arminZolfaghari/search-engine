import json
import math
import copy
from PositionalPosting import load_positional_postings_list
from Preprocess import read_data_from_file, preprocess_query
from PositionalPosting import POSITIONAL_POSTINGS_LIST_FILE_WITHOUT_STOP_WORDS, \
    POSITIONAL_POSTINGS_LIST_FILE_WITH_STOP_WORDS, DF_WITH_STOP_WORDS

VECTORS_WITHOUT_STOP_WORDS_FILE = "./vectors_without_stop_words.json"

# positional_postings_lists_with_stop_words = load_positional_postings_list(POSITIONAL_POSTINGS_LIST_FILE_WITH_STOP_WORDS)
positional_postings_list_without_stop_words = load_positional_postings_list(
    POSITIONAL_POSTINGS_LIST_FILE_WITHOUT_STOP_WORDS)

positional_postings_lists = positional_postings_list_without_stop_words
number_of_all_documents = len(read_data_from_file(DF_WITH_STOP_WORDS))


def calculate_term_freq_weight(term_freq_dict):
    """
    this function calculate tf(t, d) = (1 + log(f(t, d)))
    raw frequency => weight frequency
    :param term_freq_dict:
    :return: term_freq_dict_weight
    """
    term_freq_dict_weight = {}
    for term, raw_freq in term_freq_dict.items():
        term_freq_dict_weight[term] = 1 + math.log10(raw_freq)

    return term_freq_dict_weight


def calculate_inverse_document_frequency(term):
    """
    this function calculate idf(t) = log(N / dft)
    :param term:
    :return: idf
    """
    # dft
    document_freq_of_term = positional_postings_lists[term]["unique_documents_frequency"]
    # N
    number_of_documents = len(positional_postings_lists)
    inverse_document_frequency = math.log10(number_of_documents / document_freq_of_term)

    return inverse_document_frequency


def calculate_doc_length(term_freq_list):
    """
    calculate total squares of term frequency list
    calculate square roots (radical)
    :param term_freq_list:
    :return: doc length
    """
    sum = 0
    for number in term_freq_list:
        sum += number ** 2

    return sum ** 0.5


def calculate_term_freq_normalize(term_freq_dict):
    """
    term frequency wight / doc length
    :param term_freq_dict:
    :return: term_freq_dict_normalized
    """
    term_freq_list = list(term_freq_dict.values())
    doc_length = calculate_doc_length(term_freq_list)
    term_freq_dict_normalized = {k: v / doc_length for k, v in term_freq_dict.items()}

    return term_freq_dict_normalized


def calculate_final_weight(type, term_freq_raw):
    """
    input is term frequency - raw (tf-raw)
    calculate term frequency wight (tf-wt)
    for query calculate inverse document frequency (idf)
    then normalized weight
    :param type:
    :param term_freq_raw:
    :return: final_weight
    """
    # term frequency - weight (tf-wt)
    term_freq_weight_dict = calculate_term_freq_weight(term_freq_raw)

    # final weight: tf-wt * idf
    final_term_freq_weight = copy.deepcopy(term_freq_weight_dict)
    if type == "query":
        for term, freq_weight in term_freq_weight_dict.items():
            final_term_freq_weight[term] = freq_weight * calculate_inverse_document_frequency(term)

    # term frequency normalized
    return calculate_term_freq_normalize(final_term_freq_weight)


def create_vectors_list_from_postings_lists():
    vectors_list = {}
    for i in range(number_of_all_documents):  # i is number of document
        term_freq_dict = {}
        for term, termInfo in positional_postings_lists.items():
            if str(i) in termInfo["document_frequency_dict"]:
                # term frequency - raw (tf-raw)
                term_freq_dict[term] = termInfo["document_frequency_dict"][str(i)]
        vectors_list[i] = calculate_final_weight("document", term_freq_dict)

    return vectors_list


def save_vectors_list(vectors_list):
    with open("./vectors_without_stop_words.json", 'w', encoding='utf-8') as fp:
        json.dump(vectors_list, fp, sort_keys=True, indent=4, ensure_ascii=False)


def load_vectors_list(file_name):
    with open(file_name, 'r', encoding='utf-8') as fp:
        vectors_list = json.load(fp)

    return vectors_list


def create_query_vector(query_string, remove_stop_words_flag, stem_flag):
    query_tokens_dict = preprocess_query(query_string, "positional", remove_stop_words_flag, stem_flag)

    # term frequency - raw (tf-raw)
    query_term_freq_dict = {}
    for term, positional_index in query_tokens_dict.items():
        query_term_freq_dict[term] = len(positional_index)

    # final term frequency
    query_final_term_frequency = calculate_final_weight("query", query_term_freq_dict)

    return query_final_term_frequency


vectors_without_stop_words = load_vectors_list(VECTORS_WITHOUT_STOP_WORDS_FILE)

if __name__ == "__main__":
    v = create_vectors_list_from_postings_lists()
    save_vectors_list(v)
