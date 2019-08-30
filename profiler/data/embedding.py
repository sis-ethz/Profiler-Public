from __future__ import unicode_literals, division
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import OneHotEncoder
from gensim.models.fasttext import FastText
from scipy.spatial.distance import cosine
from profiler.globalvar import *
import numpy as np
import pandas as pd
import logging, os


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_cos(vec):
    return cosine(vec[0], vec[1])


class OneHotModel(object):
    def __init__(self, data):
        """
        :param corpus:
        :param dim:
        :param a: constant used for SIF embedding
        """
        self.encoder = self.build_vocab(data)


    def build_vocab(self, data):
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(data)
        return enc

    def get_embedding(self, data):
        return self.encoder.transform(data).toarray()


class SIF(object):

    def __init__(self, env, config, data, attr):
        """
        :param corpus:
        :param dim:
        :param a: constant used for SIF embedding
        """
        self.env = env
        self.config = config
        self.vec, self.vocab = self.load_vocab(data, attr)

    def load_vocab(self, data, attr):
        if not self.config['load']:
            # build vocab
            vec, vocab = self.build_vocab(data, attr)
        else:
            path = os.path.join(self.config['path'], attr)
            vec = np.load(path+'vec.npy', allow_pickle=True)
            unique_cells = np.load(path+'vocab.npy', allow_pickle=True)
            vocab = pd.DataFrame(data=unique_cells, columns=['word']).reset_index().set_index('word')
        return vec, vocab

    def build_vocab(self, data, attr):
        # tokenize cell
        logger.info('[%s] tokenize cell'%attr)
        corpus = [self.config['tokenizer'](i) for i in data]
        max_length = max([len(s) for s in corpus])

        # train language model
        logger.info('[%s] train language model'%attr)
        wv = LocalFasttextModel(self.env, self.config, corpus)

        # compute weights
        logger.info('[%s] compute weights'%attr)
        all_words = np.hstack(corpus)
        unique, counts = np.unique(all_words, return_counts=True)
        freq = counts / len(all_words)
        weight = self.config['a'] / (self.config['a'] + freq)

        # obtain word vector
        logger.info('[%s] create vector map'%attr)
        vec = wv.get_array_vectors(unique)
        word_vocab = pd.DataFrame(list(zip(unique, list(range(len(all_words))))),
                                  columns=['word', 'idx']).set_index('word')

        def get_cell_vector(cell):
            cell = self.config['tokenizer'](cell)
            idx = word_vocab.loc[cell, 'idx'].values
            v = vec[idx]
            if len(cell) == 1:
                return v
            w = weight[idx].reshape(1, -1)
            return list(np.matmul(w, v)/np.sum(w))
        # compute embedding for each cell
        if max_length == 1:
            unique_cells = unique
        else:
            unique_cells = np.unique(data)
            vec = np.array(list(map(get_cell_vector, unique_cells))).squeeze()
        vocab = pd.DataFrame(data=unique_cells, columns=['word']).reset_index().set_index('word')
        vocab.loc[np.nan, 'index'] = vec.shape[0]
        vec = np.vstack((vec, [-1]*vec.shape[1]))

        # (optional) save model
        if self.config['save']:
            path = os.path.join(self.config['path'], attr)
            logger.info('[%s] save vec and vocab'%attr)
            np.save(path+'vec', vec)
            np.save(path+'vocab', unique_cells)
        return vec, vocab

    def get_embedding(self, array):
        idxs = self.vocab.loc[array].values
        vecs = self.vec[idxs, :]
        return vecs


class FT(object):

    def __init__(self, env, config, data, attr):
        """
        :param corpus:
        :param dim:
        :param a: constant used for SIF embedding
        """
        self.env = env
        self.config = config
        self.vec, self.vocab = self.load_vocab(data, attr)

    def load_vocab(self, data, attr):
        if not self.config['load']:
            # build vocab
            vec, vocab = self.build_vocab(data, attr)
        else:
            path = os.path.join(self.config['path'], attr)
            vec = np.load(path+'vec.npy', allow_pickle=True)
            unique_cells = np.load(path+'vocab.npy', allow_pickle=True)
            vocab = pd.DataFrame(data=unique_cells, columns=['word']).reset_index().set_index('word')
            vocab['index'] = vocab['index'].astype(int)
        return vec, vocab

    def build_vocab(self, data, attr):
        # tokenize cell
        logger.info('[%s] tokenize cell'%attr)
        corpus = [self.config['tokenizer'](i) for i in data]
        max_length = max([len(s) for s in corpus])

        # train language model
        logger.info('[%s] train language model'%attr)
        wv = LocalFasttextModel(self.env, self.config, corpus)

        # compute weights
        logger.info('[%s] compute weights'%attr)
        all_words = np.hstack(corpus)
        unique, counts = np.unique(all_words, return_counts=True)

        # obtain word vector
        logger.info('[%s] create vector map'%attr)
        vec = wv.get_array_vectors(unique)
        word_vocab = pd.DataFrame(list(zip(unique, list(range(len(all_words))))),
                                  columns=['word', 'idx']).set_index('word')

        def get_cell_vector(cell, max_length):
            cell = self.config['tokenizer'](cell)
            idx = word_vocab.loc[cell, 'idx'].values
            if not self.config['concate']:
                v = vec[idx].reshape(len(cell), len(vec[0]))
                return list(np.sum(v, axis=0)/len(cell))
            else:
                vectors = vec[idx]
                v = np.zeros((max_length * len(vec[0]), ))
                v[0:len(cell)*len(vec[0])] = vectors.reshape((-1,))
                return list(v)

        # compute embedding for each cell
        if max_length == 1:
            unique_cells = unique
        else:
            unique_cells = np.unique(data)
            vec = np.array(list(map(lambda x: get_cell_vector(x, max_length), unique_cells))).squeeze()

        vocab = pd.DataFrame(data=unique_cells, columns=['word']).reset_index().set_index('word')
        vocab.loc[np.nan, 'index'] = vec.shape[0]
        # IMPORTANT: convert index to integer instead of float
        vocab['index'] = vocab['index'].astype(int)
        vec = np.vstack((vec, [-1]*vec.shape[1]))

        # (optional) save model
        if self.config['save']:
            path = os.path.join(self.config['path'], attr)
            logger.info('[%s] save vec and vocab'%attr)
            np.save(path+'vec', vec)
            np.save(path+'vocab', unique_cells)
        return vec, vocab

    def get_embedding(self, array):
        idxs = self.vocab.loc[array].values
        vecs = self.vec[idxs, :]
        return vecs


class LocalFasttextModel(object):

    def __init__(self, env, config, corpus):
        self.model = FastText(size=config['dim'], window=config['window'], min_count=1, batch_words=config['batch_words'])
        self.model.build_vocab(sentences=corpus)
        self.model.train(sentences=corpus, total_examples=self.model.corpus_count, epochs=config['epochs'],
                         seed=env['seed'])
        self.dim = config['dim']

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
        # configuration used for training language model
        self.param = {
            'dim': 128,
            'type': ATTRIBUTE_EMBEDDING,
            'tokenizer': lambda x: x.split(),
            'a': 1e-6,
            'path': '',
            'save': False,
            'load': False,
            'batch_words': 100,
            'window': 3,
            'epochs': 100,
            "mode": "ft",
            "concate": True,
        }

    def train(self, **kwargs):
        self.param.update(kwargs)

        if not self.param['load']:
            if not os.path.exists(self.param['path']):
                os.makedirs(self.param['path'])

        if self.param['mode'] == "sif":
            mode = SIF
        else:
            mode = FT

        if self.param['type'] == ATTRIBUTE_EMBEDDING:
            self.models = {}
            to_embed = self.ds.to_embed()
            if self.env['workers'] > 1:
                pool = ThreadPoolExecutor(self.env['workers'])
                for i, model in enumerate(pool.map(lambda attr: mode(self.env, self.param, self.ds.df[attr], attr=attr),
                                                   to_embed)):
                    self.models[to_embed[i]] = model
            else:
                for attr in to_embed:
                    self.models[attr] = mode(self.env, self.param, self.ds.df[attr], attr=attr)

        elif self.param['type'] == PRETRAINED_EMBEDDING:
            raise Exception("NOT IMPLEMENTED")
        else:
            raise Exception("[%s] is not a valid embedding type!" % self.param['type'])

    def get_embedding(self, array, attr=None):
        # handle nan
        if self.embedding_type != ATTRIBUTE_EMBEDDING:
            return self.models[0].get_array_vectors(array)
        return self.models[attr].get_embedding(array)
