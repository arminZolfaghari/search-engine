import json
import pandas as pd
import ast
from Preprocess import get_data_frame_after_preprocess

DF_WITH_STOP_WORDS = "./df_after_preprocess_with_stop_words.xlsx"
DF_WITHOUT_STOP_WORDS = "./df_after_preprocess_without_stop_words.xlsx"
POSITIONAL_POSTINGS_LIST_FILE_WITH_STOP_WORDS = "./positional_postings_lists_with_stop_words.json"
POSITIONAL_POSTINGS_LIST_FILE_WITHOUT_STOP_WORDS = "./positional_postings_lists_without_stop_words.json"
positional_postings_lists = {}


class PositionalPosting:
    def __init__(self):
        self.frequency_in_all_documents = 0
        self.document_index_dict = {}
        self.document_frequency_dict = {}
        self.unique_documents_frequency = 0

    def toJson(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)


# def load_df_after_preprocess(file_name):
#     data_frame = pd.read_excel(file_name, sheet_name='Sheet1', usecols="B,C,D")
#     return data_frame


def update_positional_postings_list(doc_id, tokens_in_document):
    for token, indexes in tokens_in_document.items():
        if token in positional_postings_lists:
            positional_posting = positional_postings_lists[token]
            positional_posting.frequency_in_all_documents += len(indexes)
            positional_posting.document_index_dict[doc_id] = indexes
            positional_posting.document_frequency_dict[doc_id] = len(indexes)
            positional_posting.unique_documents_frequency += 1
        else:
            new_positional_posting = PositionalPosting()
            new_positional_posting.frequency_in_all_documents += len(indexes)
            new_positional_posting.document_index_dict[doc_id] = indexes
            new_positional_posting.document_frequency_dict[doc_id] = len(indexes)
            new_positional_posting.unique_documents_frequency += 1
            positional_postings_lists[token] = new_positional_posting


def create_positional_postings_lists(data_frame):
    for doc_id, row in data_frame.iterrows():
        tokens_in_document = row['tokens']
        update_positional_postings_list(doc_id, tokens_in_document)

    return positional_postings_lists


def save_positional_postings_lists():
    positional_postings_lists_json = {}
    for term, postings_list in positional_postings_lists.items():
        # print(term)
        positional_postings_lists_json[term] = postings_list.__dict__
    # print(positional_postings_lists_json)
    with open(POSITIONAL_POSTINGS_LIST_FILE_WITHOUT_STOP_WORDS, 'w', encoding='utf-8') as fp:
        json.dump(positional_postings_lists_json, fp, sort_keys=True, indent=4, ensure_ascii=False)


def load_positional_postings_list(file_name):
    with open(file_name, 'r', encoding='utf-8') as fp:
        positional_postings_lists_json = json.load(fp)
        # print(positional_postings_list_json)
        global positional_postings_lists
        positional_postings_lists = positional_postings_lists_json
        # print(type(positional_postings_list))
    return positional_postings_lists


if __name__ == "__main__":
    print("in p p")
    # data_frame_after_preprocess = read_data_from_file(NEW_FILE_NAME)
    # print(type(data_frame_after_preprocess['tokens'][0]))
    data_frame_after_preprocess = get_data_frame_after_preprocess()
    print(data_frame_after_preprocess)
    create_positional_postings_lists(data_frame_after_preprocess)
    save_positional_postings_lists()
