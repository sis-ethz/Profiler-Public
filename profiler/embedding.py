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

    def __init__(self, env, data, dim, tokenizer, a=1e-3, load=False, path='', save=True):
        """

        :param corpus:
        :param dim:
        :param a: constant used for SIF embedding
        """
        self.a = a
        self.dim = dim
        self.workers = env['workers']

        if load:
            # compute weights
            corpus = [tokenizer(row) for row in data]

            # train language model
            self.wv = LocalFasttextModel(env, corpus, dim)
            self.weights, self.wordvecs = self.build_vocab(corpus, wv)
            if save:
                self.save_model(path)
        else:
            self.weights = pd.read_csv(path+'weight.csv', index_col='word')
            self.wordvecs = pd.read_csv(path+'vec.csv', index_col='word')

    def build_vocab(self, corpus, wv):
        # compute weights
        all_words = np.hstack(corpus)
        unique, counts = np.unique(all_words, return_counts=True)
        freq = counts / len(all_words)
        weight = self.a / (self.a + freq)
        weights = pd.DataFrame(list(zip(unique, weight)), columns=['word', 'weight']).set_index('word')
        # no need to handle null since it will be handled in comparison
        # handle padding
        # weights.loc['_padding_'] = np.zeros((self.dim,))
        
        # obtain word vector 
        wordvecs = pd.DataFrame(np.hstack([unique.reshape(-1,1), wv.get_array_vectors(unique)])).set_index(0)
        wordvecs.index.name = 'word'
        return weights, wordvecs
    
    def save_model(self, path):
        self.weights.to_csv(path+'weight.csv')
        self.wordvecs.to_csv(path+'vec.csv')
    
    def get_weights(self, words):
        return self.weights.loc[words].values
    
    def get_wv(self, words):
        return self.wordvecs.loc[words].values

    def get_cell_vector(self, cell):
        if isinstance(cell, str):
            cell = self.tokenizer(cell)
        w = self.get_weights(cell).transpose()
        v = self.get_wv(cell)
        return np.matmul(w, v)/np.sum(w)

    def get_array_vectors(self, array):
        # TODO: add parallellization option
        return np.array(list(map(self.get_cell_vector, array))).squeeze()


class LocalFasttextModel(object):

    def __init__(self, env, corpus, dim):
        self.model = FastText(size=dim, window=3, min_count=1, batch_words=100)
        self.model.build_vocab(sentences=corpus)
        self.model.train(sentences=corpus, total_examples=self.model.corpus_count, epochs=self.model.epochs,
                         seed=env['seed'])
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

    def train(self, embedding_size, embedding_type, tokenizer=lambda x: x.split(), path='', save=True, load=False):
        self.embedding_type = embedding_type
        if self.embedding_type == ATTRIBUTE_EMBEDDING:
            self.models = {}
            to_embed = self.ds.to_embed()
            if self.env['workers'] > 1:
                pool = ThreadPoolExecutor(self.env['workers'])
                for i, model in enumerate(pool.map(lambda x: SIF(self.env, self.ds.df[x], dim=embedding_size,
                                                                 tokenizer=tokenizer), to_embed)):
                    self.models[to_embed[i]] = model
            else:
                for attr in to_embed:
                    self.models[attr] = SIF(self.env, self.ds.df[attr], dim=embedding_size, tokenizer=tokenizer, 
                        path=os.path.join(path+attr), load=load, save=save)
        elif self.embedding_type == ONE_HOT_EMBEDDING:
            raise Exception("NOT IMPLEMENTED")
            # self.models = [OneHotEncoderModel(self, source_data)]
        elif self.embedding_type == PRETRAINED_EMBEDDING:
            raise Exception("NOT IMPLEMENTED")
        else:
            raise Exception("[%s] is not a valid embedding type!" % embedding_type)
        #self.dim = self.models[0].dim

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




