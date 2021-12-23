import json
import numpy as np
from numpy.linalg import norm
from TrainModel import w2v_model
from VectorSpace import vectors_without_stop_words, calculate_final_weight
from Preprocess import preprocess_query
from Score import sort_doc_score_dict

DOCS_EMBEDDING_FILE = "./docs_embedding.json"


def final_search_result_in_embedding(query_vector, top_k):
    """
    return final (best) top k list
    :param query_vector:
    :param top_k:
    :return: top_k_list
    """
    sorted_document_score_with_query = calculate_document_score_with_query_in_embedding(query_vector)
    return list(sorted_document_score_with_query.items())[0: top_k]


def calculate_document_score_with_query_in_embedding(query_vector):
    doc_id_score_dict = {}
    for doc_vector in docs_embedding:
        score = calculate_similarity(query_vector, doc_vector)
        doc_id_score_dict[docs_embedding.index(doc_vector)] = score

    sorted_document_id_score = sort_doc_score_dict(doc_id_score_dict)
    return sorted_document_id_score


def create_docs_embedding():
    # print(len(vectors_without_stop_words["0"]))
    # print(len(w2v_model.wv.vo))
    docs_embedding = []
    for doc, doc_info in vectors_without_stop_words.items():
        doc_vector = np.zeros(300)
        weights_sum = 0
        for token, weight in doc_info.items():
            doc_vector += w2v_model.wv[token] * weight
            weights_sum += weight

        docs_embedding.append(doc_vector / weights_sum)

    return docs_embedding


def save_docs_embedding(docs_embedding, file_name):
    print(docs_embedding)
    print(type(docs_embedding))
    print(type(docs_embedding[0]))
    docs_embedding_list = []
    for doc in docs_embedding:
        print(doc)
        docs_embedding_list.append(list(doc))

    print(1111111111111)
    print(docs_embedding_list)

    with open(file_name, 'w', encoding='utf-8') as fp:
        json.dump(docs_embedding_list, fp, sort_keys=True, indent=4, ensure_ascii=False)


def load_docs_embedding(file_name):
    with open(file_name, 'r', encoding='utf-8') as fp:
        docs_embedding = json.load(fp)

    return docs_embedding


def calculate_similarity(doc1, doc2):
    similarity_score = np.dot(doc1, doc2) / (norm(doc1) * norm(doc2))

    return (similarity_score + 1) / 2


def create_query_embedding(query_string):
    query_tokens_dict = preprocess_query(query_string, "positional", True, True)

    # term frequency - raw (tf-raw)
    query_term_freq_dict = {}
    for term, positional_index in query_tokens_dict.items():
        query_term_freq_dict[term] = len(positional_index)

    # final term frequency
    query_final_term_frequency = calculate_final_weight("query", query_term_freq_dict)
    print(query_final_term_frequency)

    query_vector_embedding = np.zeros(300)
    weight_sum = 0
    for token, weight in query_final_term_frequency.items():
        query_vector_embedding += w2v_model.wv[token] * weight
        weight_sum += weight
    query_vector_embedding /= weight_sum

    return query_vector_embedding


# docs_embedding = create_docs_embedding()
# save_docs_embedding(docs_embedding, DOCS_EMBEDDING_FILE)
docs_embedding = load_docs_embedding(DOCS_EMBEDDING_FILE)
# print(docs_embedding)


if __name__ == "__main__":
    query_vector = create_query_embedding("تیم سپاهان در لیگ برتر")
    print(final_search_result_in_embedding(query_vector, 10))

