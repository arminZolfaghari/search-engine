from Preprocess import preprocess_word, df_before_preprocess
import json
from collections import OrderedDict
from VectorSpace import create_query_vector
from Score import final_search_result
from Analysis import check_doc_content_with_query
from ANNEmbedding import final_search_result_in_embedding, create_query_vector_embedding

POSITIONAL_POSTINGS_LIST_FILE_WITH_STOP_WORDS = "./positional_postings_lists_with_stop_words.json"
POSITIONAL_POSTINGS_LIST_FILE_WITHOUT_STOP_WORDS = "./positional_postings_lists_without_stop_words.json"


def print_result(query_vector, doc_id_score_list):
    for doc_info in doc_id_score_list:
        doc_id, doc_score = doc_info[0], doc_info[1]
        print(f"doc id: {doc_id}, score: {doc_score}")
        title = df_before_preprocess['title'][int(doc_id)]
        print(f"title: {title}")
        related_sentences_with_query_in_doc = check_doc_content_with_query(query_vector, doc_id)
        print(f"---------- related sentences in this document ----------")
        for sentence in related_sentences_with_query_in_doc:
            print(f"sentence {related_sentences_with_query_in_doc.index(sentence) + 1}: {sentence}")
        print('*********************************************************************')
        print('*********************************************************************')


def search_word(word):
    # preprocess
    new_word = preprocess_word(word)
    inf_about_word = positional_postings_lists[new_word]
    number_of_documents = inf_about_word['unique_documents_frequency']
    document_frequency_dict = inf_about_word['document_frequency_dict']
    doc_id_list = [*document_frequency_dict]
    print("Result: " + str(number_of_documents) + " documents")
    print_result(doc_id_list)


def find_docs_intersection(words_list):
    docs_id_lists = []
    for word in words_list:
        inf_about_word = positional_postings_lists[word]
        document_frequency_dict = inf_about_word['document_frequency_dict']
        docs_id_list = [*document_frequency_dict]
        docs_id_lists.append(docs_id_list)

    docs_id_intersection_list = docs_id_lists[0]
    for i in range(1, len(docs_id_lists)):
        docs_id_intersection_list = list(set(docs_id_intersection_list) & set(docs_id_lists[i]))

    return docs_id_intersection_list


def check_words_index(words_list, docs_id_intersection):
    result_docs_id_lists = []
    for doc_id in docs_id_intersection:
        inf_about_word = positional_postings_lists[words_list[0]]

        for index in inf_about_word["document_index_dict"][doc_id]:

            flag = True
            for i in range(1, len(words_list)):
                inf_about_next_word = positional_postings_lists[words_list[i]]
                if int(int(index) + i) not in inf_about_next_word["document_index_dict"][doc_id]:
                    flag = False
                    break

            if flag:
                if doc_id not in result_docs_id_lists:
                    result_docs_id_lists.append(doc_id)

    return result_docs_id_lists


def create_list_from_dict(result_docs_id_dict):
    result_docs_id_list = []
    for key, values in sorted(result_docs_id_dict.items(), reverse=True):
        for value in sorted(values):
            if value not in result_docs_id_list:
                result_docs_id_list.append(value)

    return result_docs_id_list


def search_words(words):
    # preprocess
    new_words = []
    for word in words:
        new_words.append(preprocess_word(word))

    result_docs_id_dict = {}
    for i in range(len(new_words) - 1, -1, -1):
        docs_id_lists = []
        for j in range(0, len(new_words) - i):
            docs_id_intersection = find_docs_intersection(new_words[j:i + j + 1])
            if len(docs_id_intersection) >= 1:
                docs_id_lists = list(
                    set(docs_id_lists + check_words_index(new_words[j:i + j + 1], docs_id_intersection)))
        result_docs_id_dict[i + 1] = docs_id_lists

    for key, value in result_docs_id_dict.items():
        print(f'We have {len(value)} document(s) that have {key} words of the user query!')
    print("*******************************************************************************************")
    print(result_docs_id_dict)

    result_docs_id_list = create_list_from_dict(result_docs_id_dict)
    print_result(result_docs_id_list)


if __name__ == "__main__":
    # load_positional_postings_list(POSITIONAL_POSTINGS_LIST_FILE_WITH_STOP_WORDS)
    # arr = ["دانشگاه", "صنعتی", "امیرکبیر"]
    # search_words(arr)
    query_sentence = ""
    while query_sentence != "-1":
        print("Enter query:")
        query_sentence = input()
        query_vector = create_query_vector(query_sentence, False, True)
        print(query_vector)
        final_doc_score_list_in_tf_idf = final_search_result(query_vector, 10, True, False)
        print("########## tf - idf ##########")
        print(final_doc_score_list_in_tf_idf)
        print_result(query_vector, final_doc_score_list_in_tf_idf)

        print("########## W2V model (my model) ##########")
        query_vector_embedding = create_query_vector_embedding(query_sentence, "my model")
        final_doc_score_list_in_w2v = final_search_result_in_embedding(query_vector_embedding, 10)
        print(final_doc_score_list_in_w2v)
        print_result(query_vector, final_doc_score_list_in_w2v)

        print("########## W2V model (hazm model) ##########")
        query_vector_embedding = create_query_vector_embedding(query_sentence, "hazm model")
        final_doc_score_list_in_w2v = fi