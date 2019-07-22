import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import time
from nltk.tokenize import word_tokenize
import numpy as np


def train(inp="wiki2.he.text", out_model="wiki.he.word2vec.model"):
    start = time.time()
    model = Word2Vec(LineSentence(inp), sg = 1, # 0=CBOW , 1= SkipGram
                     size=100, window=5, min_count=5, workers=multiprocessing.cpu_count())
    model.init_sims(replace=True)
    print(time.time()-start)
    model.save(out_model)


def get_model(model="wiki.he.word2vec.model"):
    model = Word2Vec.load(model)
    return model


def get_word_vector(model, word):
    return model.wv[word]


def get_post_vector(our_word2vec_model, wiki_word2vec_model, post):
    post = word_tokenize(post)
    # remove out-of-vocabulary words
    postEmbedding = []
    for word in post:
        if word in our_word2vec_model.wv.vocab:
            postEmbedding.append(our_word2vec_model.wv[word])
        elif word in wiki_word2vec_model.wv.vocab:
            postEmbedding.append(wiki_word2vec_model.wv[word])

    if len(postEmbedding) == 0:
        return np.zeros((1, 100))
    return np.mean(postEmbedding, axis=0)
