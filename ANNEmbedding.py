import ast
import json
import multiprocessing
from Preprocess import read_data_from_file
from numpy.linalg import norm
import time
import numpy as np

from gensim.models import Word2Vec


