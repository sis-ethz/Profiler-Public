from __future__ import unicode_literals, division

import logging
import tempfile

import numpy as np
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from gensim.models.fasttext import FastText
from scipy.spatial.distance import cosine
from globalvar import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_cos(vec):
    return cosine(vec[0],vec[1])


class OneHotEncoderModel(object):

    def __init__(self, parent, data):
        self.label_encoder = LabelEncoder()
        integer_encoded = self.label_encoder.fit_transform(data)
        self.encoder = OneHotEncoder(sparse=False, categories='auto', handle_unknown='ignore')
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        self.encoder.fit(integer_encoded)
        self.shape = self.get_word_vector(data[0]).shape
        self.dim = self.shape[1]
        logger.info("Attr Embedding Size: {}".format(self.dim))

    def get_word_vector(self, word):
        # contains a single sample
        labeled = self.label_encoder.transform([word]).reshape(1, -1)
        encoded = self.encoder.transform(labeled)
        # except ValueError:
        #     encoded = np.zeros(self.dim)
        return encoded

    def get_array_vectors(self, array):
        labeled = self.label_encoder.transform([array]).reshape(1, -1)
        encoded = self.encoder.transform(labeled)
        # except ValueError:
        #     encoded = np.zeros(self.dim)
        return encoded

    def get_wv(self):
        return pd.Series(self.encoder.categories_[0],
                         index=[x.encode('UTF8') for x in self.label_encoder.classes_]).to_dict()


class LocalFasttextModel(object):

    def __init__(self, parent, data, dim):
        self.embedding = parent
        self.model = FastText(size=dim, window=3, min_count=1, batch_words=100)
        self.model.build_vocab(sentences=data)
        self.model.train(sentences=data, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        self.dim = dim

    def get_word_vector(self, word):
        return self.model.wv[word]

    def get_array_vectors(self, array):
        return self.model.wv[array]

    def get_wv(self):
        return self.model.wv


class Embedding(object):

    def __init__(self, parent, embedding_file="local.bin", embedding_type=None, embedding_size=128):
        self.model = None
        self.dataEngine = parent
        self.embedding_file = embedding_file
        self.embedding_type = embedding_type

        source_data = self.dataEngine.get_embedding_source(embedding_type=self.embedding_type)

        if self.embedding_type == ONE_HOT_EMBEDDING:
            self.model = OneHotEncoderModel(self, source_data)
        elif self.embedding_type == PRETRAINED_EMBEDDING:
            raise Exception("NOT IMPLEMENTED")
        elif self.embedding_type == ATTRIBUTE_EMBEDDING:
            self.models = {}
            for attr in source_data:
                self.models[attr] = LocalFasttextModel(self, source_data[attr], dim=embedding_size)
        else:
            self.model = LocalFasttextModel(self, source_data, dim=embedding_size)
        # one hot encoding does not take embedding_size as an input!
        if self.model:
            self.dim = self.model.dim
        else:
            self.dim = self.models.values()[0].dim

    def get_word_vector(self, word, attr=None):
        if word == self.dataEngine.param['nan']:
            return np.array([np.nan]*self.dim)
        if self.embedding_type != ATTRIBUTE_EMBEDDING:
            return self.model.get_word_vector(word)
        return self.models[attr].get_word_vector(word)

    def get_array_vectors(self, array, attr=None):
        # handle nan
        if self.embedding_type != ATTRIBUTE_EMBEDDING:
            return self.model.get_array_vectors(array)
        return self.models[attr].get_array_vectors(array)

    def get_pair_distance(self, a, b, attr=None):
        nan = (a == self.dataEngine.param['nan']) | (b == self.dataEngine.param['nan'])
        nan_id = np.array(range(a.shape[0]))[nan]
        vec1 = self.get_array_vectors(a[~nan], attr=attr)
        vec2 = self.get_array_vectors(b[~nan], attr=attr)

        #p = Pool(self.dataEngine.param['workers'])
        #sim = p.map(get_cos, zip(vec1, vec2))
        sim = [cosine(v1, v2) for v1, v2 in tqdm(zip(vec1, vec2))]
        for nid in nan_id:
            sim = np.insert(sim, nid, np.nan)
        return sim




