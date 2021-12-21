import json
import math
from PositionalPosting import load_positional_postings_list
from VectorSpace import load_vectors_list, positional_postings_lists

CHAMPIONS_LIST_FILE = "./champions_list.json"
POSITIONAL_POSTINGS_LIST_FILE_WITH_STOP_WORDS = "./positional_postings_lists_with_stop_words.json"
VECTORS_WITHOUT_STOP_WORDS_LIST = "./vectors_without_stop_words.json"

vectors_list = load_vectors_list(VECTORS_WITHOUT_STOP_WORDS_LIST)


def final_search_result(query_vector, top_k, index_elimination_flag, champions_list_flag):
    """
    return final top k list
    :param query_vector:
    :param champions_list_flag:
    :param index_elimination_flag:
    :param top_k:
    :return: top_k_list
    """
    sorted_document_score_with_query = calculate_document_score_with_query(query_vector, index_elimination_flag,
                                                                           champions_list_flag)
    return list(sorted_document_score_with_query.items())[0: top_k]


def calculate_document_score_with_query(query_vector, index_elimination_flag=False, champions_list_flag=False):
    if index_elimination_flag:
        related_documents_list = index_elimination(query_vector)

    if champions_list_flag:
        related_documents_list = get_related_documents_with_champions(query_vector)

    document_id_score_dict = {}
    for document_id in related_documents_list:
        score = cos_similarity(vectors_list[document_id], query_vector)
        document_id_score_dict[document_id] = score

    sorted_document_id_score_dict = sort_doc_score_dict(document_id_score_dict)
    return sorted_document_id_score_dict


def sort_doc_score_dict(document_id_score_dict):
    sorted_doc_score_dict = dict(sorted(document_id_score_dict.items(), key=lambda x: x[1], reverse=True))

    return sorted_doc_score_dict


def cos_similarity(vector_a, vector_b):
    """
    calculate cos similarity of 2 normalized vectors
    :param vector_a:
    :param vector_b:
    :return: cos_similarity_value
    """
    intersection_list = vector_a.keys() & vector_b.keys()
    cos_similarity_value = 0
    for term in intersection_list:
        cos_similarity_value += vector_a[term] * vector_b[term]

    return cos_similarity_value


def index_elimination(query_vector):
    documents_list_after_index_elimination = []
    for term in query_vector:
        documents_list = positional_postings_lists[term]["document_frequency_dict"].keys()
        # union of documents list has at least one term of the query
        print(documents_list)
        print(type(documents_list))
        documents_list_after_index_elimination = list(
            set(list(documents_list) + documents_list_after_index_elimination))

    return documents_list_after_index_elimination


def get_related_documents_with_champions(query_vector):
    related_documents = []
    for term in query_vector:
        documents_list = term_champions_dict[term]
        related_documents = list(set(documents_list + related_documents))

    return related_documents


def create_champions_list():
    """
    for increase performance, we create champions list for all term in dictionary
    if term exists more than 10 documents, we save 60% of documents that highest score in champions list
    else we save all documents in champions list
    :return: term_champions_dict
    """
    term_champions_dict = {}
    for term, termInfo in positional_postings_lists.items():
        document_frequency_dict = termInfo["document_frequency_dict"]  # key is document id, value is frequency

        sorted_document_frequency_dict = sort_doc_score_dict(document_frequency_dict)
        if len(sorted_document_frequency_dict) <= 10:
            term_champions_dict[term] = list(sorted_document_frequency_dict.keys())
        else:
            champion_list_len = math.ceil(len(sorted_document_frequency_dict) * 6 / 10)
            term_champions_dict[term] = list(sorted_document_frequency_dict.keys())[0: champion_list_len]

    return term_champions_dict


def save_champions_list(file_name, term_champions_list):
    with open(file_name, 'w', encoding='utf-8') as fp:
        json.dump(term_champions_list, fp, sort_keys=True, indent=4, ensure_ascii=False)


def load_champions_list(file_name):
    with open(file_name, 'r', encoding='utf-8') as fp:
        term_champions_dict = json.load(fp)

    return term_champions_dict


term_champions_dict = load_champions_list(CHAMPIONS_LIST_FILE)

if __name__ == "__main__":
    pass
