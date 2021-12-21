import ast
import json
import multiprocessing
from Preprocess import read_data_from_file
from numpy.linalg import norm
import time
import numpy as np
from gensim.models import Word2Vec

cores = multiprocessing.cpu_count()
FILE_NAME = "./IR1_7k_news.xlsx"
TRAINING_DATA_FILE = "./training_data.xlsx"
TRAINING_DATA_JSON = "./training_data.json"



def load_training_data(training_data_json):
    with open(training_data_json, 'r', encoding='utf-8') as fp:
        training_data_list = json.load(fp)

    return training_data_list




def save_training_data(training_data_file, training_data_json):
    training_data_list = []
    df_training_data = read_data_from_file(training_data_file)
    for doc_id, row in df_training_data.iterrows():
        training_data_list.append(ast.literal_eval(row['tokens']))

    with open(training_data_json, 'w', encoding='utf-8') as fp:
        json.dumps(training_data_list, fp, sort_keys=True, indent=4, ensure_ascii=False)


def create_w2v_model(min_count, window, vector_size, alpha, workers):
    w2v_model = Word2Vec(min_count=min_count, window=window, vector_size=vector_size, alpha=alpha, workers=workers)
    w2v_model.build_vocab(TRAINING_DATA)
    w2v_model_vocab_size = len(w2v_model.wv)
    print("vocab size ", w2v_model_vocab_size)

    start = time.time()
    w2v_model.train(TRAINING_DATA, total_examples=w2v_model.corpus_count, epochs=20)
    end = time.time()
    print("Duration: ", end - start)
    return w2v_model


def save_w2v_model(w2v_model, model_name):
    w2v_model.save(model_name)


def load_w2v_model(model_name):
    return Word2Vec.load(model_name)


TRAINING_DATA = load_training_data(TRAINING_DATA_JSON)

# w2v_300d = create_w2v_model(1, 5, 3
# 00, 0.03, cores - 1)
# save_w2v_model(w2v_300d, "w2v_300d.model")
model = load_w2v_model("w2v_300d.model")
print((model.wv.most_similar('خبرگزار')))
