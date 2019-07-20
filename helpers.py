import pathlib

from source import utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np
import os.path
import pandas as pd
from source.Embedding import word2vec as w2v

SOURCE = os.path.abspath(os.path.join(__file__, '../../'))



def get_tf_vectorizer_data(posts):
    tf_vectorizer = utils.get_model(os.path.join(SOURCE, "outputs", "tf.pkl"))
    if tf_vectorizer is None:
        tf_vectorizer = CountVectorizer(max_df=0.6, min_df=0.01, stop_words=utils.get_stop_words())
        tf_vectorizer.fit(posts)
        utils.save_model(tf_vectorizer, os.path.join(SOURCE, 'outputs', 'tf.pkl'))

    return tf_vectorizer.transform(posts)


def reduce_damnation(mat):
    svd_model = utils.get_model(os.path.join(SOURCE, "outputs", "svd.pkl"))
    if svd_model is None:
        svd_model = TruncatedSVD(n_components=1)
        svd_model.fit(mat)
        utils.save_model(svd_model, os.path.join(SOURCE, 'outputs', 'svd.pkl'))

    svd_transform = svd_model.transform(mat)
    return np.array(svd_transform).flatten()


def get_meaningful_words_tf_idf_difference(df):
    path = SOURCE + '/outputs/MeaningfulWords.pkl'
    path_object = pathlib.Path(path)
    if path_object.exists():
        return pd.read_pickle(path)
    df_neg = utils.get_abusive_df(df)
    df_pos = utils.get_no_abusive_df(df)
    posts = [' '.join(df_neg['text'].tolist()), ' '.join(df_pos['text'].tolist())]

    tfidf = utils.get_model(os.path.join(SOURCE, "outputs", "tfidf.pkl"))
    if tfidf is None:
        tfidf = TfidfVectorizer(stop_words=utils.get_stop_words(), ngram_range=(1, 2))
        tfidf.fit(posts)
        utils.save_model(tfidf, os.path.join(SOURCE, 'outputs', 'tfidf.pkl'))

    x = tfidf.transform(posts)
    x = x[0, :] - x[1, :]
    df_tf_idf = pd.DataFrame(x.toarray(), columns=tfidf.get_feature_names())
    df_tf_idf = df_tf_idf.sort_values(by=0, axis=1, ascending=False)
    df_tf_idf.to_pickle(path)
    return df_tf_idf


def get_distance_df(df, column_name, sentence, distance_type='euclidean'):
    df_offensive_distance = pd.DataFrame(columns=['id', column_name])
    df_offensive_distance['id'] = df['id'].tolist()

    m_wiki = w2v.get_model(SOURCE + "/Embedding/wiki.he.word2vec.model")
    m_our = w2v.get_model(SOURCE + "/Embedding/our.corpus.word2vec.model")

    df_offensive_distance[column_name] = df['text'].apply(
        lambda x:
        utils.calculate_distance(w2v.get_post_vector(m_our, m_wiki, x),
                                 w2v.get_post_vector(m_our, m_wiki, sentence), distance_type))
    return df_offensive_distance


def create_vectors_array(posts, m_our, m_wiki):
    matrix = np.zeros((len(posts), 100))
    for i, post in enumerate(posts):
        embedding_vector = w2v.get_post_vector(m_our, m_wiki, post)
        matrix[i] = embedding_vector
    return matrix
