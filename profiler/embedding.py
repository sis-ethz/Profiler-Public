from __future__ import unicode_literals, division
from concurrent.futures import ThreadPoolExecutor
from gensim.models.fasttext import FastText
from scipy.spatial.distance import cosine
from profiler.globalvar import *
import numpy as np
import pandas as pd
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_cos(vec):
    return cosine(vec[0], vec[1])


class SIF(object):

    def __init__(self, env, data, dim, tokenizer, a=1e-3):
        """

        :param corpus:
        :param dim:
        :param a: constant used for SIF embedding
        """
        self.a = a
        self.dim = dim
        self.workers = env['workers']

        # compute weights
        corpus = np.array(map(tokenizer, data))
        self.weights = self.compute_weights(corpus)

        # train language model
        self.wv = LocalFasttextModel(env, corpus, dim)

    def compute_weights(self, corpus):
        all_words = np.hstack(corpus)
        unique, counts = np.unique(all_words, return_counts=True)
        freq = counts / len(all_words)
        weight = self.a / (self.a + freq)
        weights = pd.DataFrame(np.hstack[unique, weight], columns=['word', 'weight']).set_index('word', inplace=True)
        # no need to handle null since it will be handled in comparison
        # handle padding
        # weights.loc['_padding_'] = np.zeros((self.dim,))
        return weights

    def get_weights(self, words):
        return self.weights.loc[words]

    def get_cell_vector(self, cell):
        if isinstance(cell, str):
            cell = [cell]
        return np.matmul(self.get_weights(cell), self.wv.get_array_vectors(cell))/len(cell)

    def get_array_vectors(self, array):
        # TODO: add parallellization option
        return np.array(map(self.get_cell_vector, array))


class LocalFasttextModel(object):

    def __init__(self, env, corpus, dim):
        self.model = FastText(size=dim, window=3, min_count=1, batch_words=100)
        self.model.build_vocab(sentences=corpus)
        self.model.train(sentences=corpus, total_examples=self.model.corpus_count, epochs=self.model.epochs,
                         workers=env['workers'], seed=env['seed'])
        self.dim = dim

    def get_word_vector(self, word):
        return self.model.wv[word]

    def get_array_vectors(self, array):
        """

        :param array: 2d array
        :return:
        """
        return self.model.wv[array]

    def get_wv(self):
        return self.model.wv


class EmbeddingEngine(object):

    def __init__(self, env, ds):
        self.env = env
        self.ds = ds
        self.embedding_type = ATTRIBUTE_EMBEDDING
        self.models = None
        self.dim = -1

    def train(self, embedding_size, embedding_type, tokenizer=lambda x: x.split()):
        self.embedding_type = embedding_type
        if self.embedding_type == ATTRIBUTE_EMBEDDING:
            self.models = {}
            for attr in self.ds.df:
                self.models[attr] = SIF(self.env, self.ds.df[attr], dim=embedding_size, tokenizer=tokenizer)
        elif self.embedding_type == ONE_HOT_EMBEDDING:
            raise Exception("NOT IMPLEMENTED")
            # self.models = [OneHotEncoderModel(self, source_data)]
        elif self.embedding_type == PRETRAINED_EMBEDDING:
            raise Exception("NOT IMPLEMENTED")
        else:
            raise Exception("[%s] is not a valid embedding type!" % embedding_type)
        self.dim = self.models[0].dim

    def get_word_vector(self, word, attr=None):
        if word == self.ds.nan:
            return np.array([np.nan]*self.dim)
        if self.embedding_type != ATTRIBUTE_EMBEDDING:
            return self.models[0].get_word_vector(word)
        return self.models[attr].get_word_vector(word)

    def get_array_vectors(self, array, attr=None):
        # handle nan
        if self.embedding_type != ATTRIBUTE_EMBEDDING:
            return self.models[0].get_array_vectors(array)
        return self.models[attr].get_array_vectors(array)

    def get_pair_distance(self, a, b, attr=None):
        nan = (a == self.ds.nan) | (b == self.ds.nan)
        vec1 = self.get_array_vectors(a[~nan], attr=attr)
        vec2 = self.get_array_vectors(b[~nan], attr=attr)
        sim = np.zeros(a.shape[0], dtype=float)
        sim[nan] = np.nan
        # TODO: improve
        #sim[~nan] = map(get_cos, zip(vec1, vec2))
        #np.dot(vec1, vec2)/(np.linalg.norm(vec1, axis=0)*np.linalg.norm(vec2, axis=0))
        sim[~nan] = [np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)) for v1, v2 in zip(vec1, vec2)]
        return sim




