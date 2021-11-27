from Preprocess import preprocess_word, read_data_from_file
import json
from collections import OrderedDict

POSITIONAL_POSTINGS_LIST_FILE_WITH_STOP_WORDS = "./positional_postings_lists_with_stop_words.json"
POSITIONAL_POSTINGS_LIST_FILE_WITHOUT_STOP_WORDS = "./positional_postings_lists_without_stop_words.json"
MAIN_FILE_NAME = "./IR1_7k_news.xlsx"


def load_positional_postings_list(file_name):
    with open(file_name, 'r', encoding='utf-8') as fp:
        positional_postings_list_json = json.load(fp)
        # print(positional_postings_list_json)
        global positional_postings_list
        positional_postings_list = positional_postings_list_json
        # print(type(positional_postings_list))
    return positional_postings_list

def print_result(doc_id_list):
    data_frame = read_data_from_file(MAIN_FILE_NAME)
    for doc_id in doc_id_list[0: 10]:
        print("doc id: ", doc_id)
        title = data_frame['title'][int(doc_id)]
        print("title: ", title)
        print('******************************************')


def search_word(word):
    # preprocess
    new_word = preprocess_word(word)
    # print(" 22 :", positional_postings_list)
    inf_about_word = positional_postings_list[new_word]
    number_of_documents = inf_about_word['unique_documents_frequency']
    document_frequency_dict = inf_about_word['document_frequency_dict']
    doc_id_list = [*document_frequency_dict]
    print("Result: " + str(number_of_documents) + " documents")
    print_result(doc_id_list)


def find_docs_intersection(words_list):
    docs_id_lists = []
    for word in words_list:
        inf_about_word = positional_postings_list[word]
        document_frequency_dict = inf_about_word['document_frequency_dict']
        docs_id_list = [*document_frequency_dict]
        docs_id_lists.append(docs_id_list)

    docs_id_intersection_list = docs_id_lists[0]
    # print(docs_id_lists)
    for i in range(1, len(docs_id_lists)):
        docs_id_intersection_list = list(set(docs_id_intersection_list) & set(docs_id_lists[i]))

    return docs_id_intersection_list


def check_words_index(words_list, docs_id_intersection):
    result_docs_id_lists = []
    for doc_id in docs_id_intersection:
        inf_about_word = positional_postings_list[words_list[0]]
        # print("doc id: ", doc_id, " word ", words_list[0], " ", inf_about_word["document_index_dict"][doc_id])
        for index in inf_about_word["document_index_dict"][doc_id]:
            # print(type(index))
            flag = True
            for i in range(1, len(words_list)):
                inf_about_next_word = positional_postings_list[words_list[i]]
                if int(int(index) + i) not in inf_about_next_word["document_index_dict"][doc_id]:
                    # print(str(int(index) + i) , " not in ", inf_about_next_word["document_index_dict"][doc_id], " word: ", words_list[i])
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
    load_positional_postings_list(POSITIONAL_POSTINGS_LIST_FILE_WITH_STOP_WORDS)
    arr = ["دانشگاه", "صنعتی", "امیرکبیر"]
    search_words(arr)