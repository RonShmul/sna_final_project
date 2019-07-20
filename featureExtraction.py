from nltk import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from source import utils
from source.Embedding import word2vec as w2v
import os.path
from . import helpers
import pathlib
SOURCE = os.path.abspath(os.path.join(__file__, '../../'))


def get_functions_dictionary():
    return {
        'tfidf': extract_tf_idf,
        'post_length': extract_post_length,
        'topics': extract_topics,
        'screamer': extract_screamer,
        'words': extract_meaningful_words_existence,
        'off_dis': extract_distance_from_offensive,
        'not_off_dis': extract_distance_from_not_offensive,
        'wmd_off': extract_wmd_offensive,
        'wmd_not_off': extract_wmd_not_offensive,
        'dis_avg_vec': extract_distance_from_avg_vector
    }


def extract_wmd_offensive(df):
    df_wmd_offensive = pd.DataFrame(columns=['id', 'wmd_off_tfidf'])
    df_wmd_offensive['id'] = df['id'].tolist()
    tf_idf_difference = helpers.get_meaningful_words_tf_idf_difference(df)
    offensive_words_tf_idf = tf_idf_difference.iloc[:, 0:20]
    offensive_words_tf_idf = list(offensive_words_tf_idf.columns.values)
    m_our = w2v.get_model(SOURCE + "/Embedding/our.corpus.word2vec.model")
    m_our.init_sims(replace=True)  # Normalizes the vectors in the word2vec class.
    df_wmd_offensive['wmd_off_tfidf'] = df['text'].apply(
        lambda x:
        float("{0:.4f}".format(m_our.wmdistance(offensive_words_tf_idf, word_tokenize(x))))
    )
    return df_wmd_offensive


def extract_wmd_not_offensive(df):
    df_wmd_not_offensive = pd.DataFrame(columns=['id', 'wmd_not_off_tfidf'])
    df_wmd_not_offensive['id'] = df['id'].tolist()
    tf_idf_difference = helpers.get_meaningful_words_tf_idf_difference(df)
    not_offensive = tf_idf_difference.iloc[:, -20:-1]
    not_offensive_words_tf_idf = list(not_offensive.columns.values)
    m_our = w2v.get_model(SOURCE + "/Embedding/our.corpus.word2vec.model")
    m_our.init_sims(replace=True)  # Normalizes the vectors in the word2vec class.
    df_wmd_not_offensive['wmd_not_off_tfidf'] = df['text'].apply(
        lambda x:
        float("{0:.4f}".format(m_our.wmdistance(not_offensive_words_tf_idf, word_tokenize(x))))
    )
    return df_wmd_not_offensive


def extract_tf_idf(df):
    posts = df['text'].tolist()

    tf_idf_model = utils.get_model(os.path.join(SOURCE + "/outputs", "tfidf.pkl"))
    if tf_idf_model is None:
        tf_idf_model = TfidfVectorizer(stop_words=utils.get_stop_words(), ngram_range=(1, 2))
        tf_idf_model.fit(posts)
        utils.save_model(tf_idf_model, os.path.join(SOURCE + '/outputs', 'tfidf.pkl'))

    tf_idf_matrix = tf_idf_model.transform(posts)

    tf_idf_dataframe = pd.DataFrame(columns=['id', 'tfidf'])
    tf_idf_dataframe['id'] = df['id'].tolist()
    tf_idf_dataframe['tfidf'] = helpers.reduce_damnation(tf_idf_matrix)
    return tf_idf_dataframe


def extract_post_length(df):
    df_length = pd.DataFrame(columns=['id', 'post_length'])
    df_length['id'] = df['id'].tolist()
    df_length['post_length'] = df['text'].apply(lambda x: len(word_tokenize(x)))
    return df_length


def extract_topics(df, num_of_topics): #todo- add dynamic number of topics
    posts = df['text'].values
    tf_transform = helpers.get_tf_vectorizer_data(posts)
    lda = utils.get_model(os.path.join(SOURCE + "/outputs", "lda.pkl"))
    if lda is None:
        lda = LatentDirichletAllocation(n_topics=3, max_iter=5, learning_method='online', learning_offset=50.,
                                        random_state=0)
        lda.fit(tf_transform)
        utils.save_model(lda, os.path.join(SOURCE + "/outputs", "lda.pkl"))

    dt_matrix = lda.transform(tf_transform)
    features = pd.DataFrame(dt_matrix, columns=['T1', 'T2', 'T3'])
    features['writer'] = df['writer'].tolist()
    return features


def extract_screamer(df):
    df_screamer = pd.DataFrame(columns=['id', 'screamer'])
    df_screamer['id'] = df['id'].tolist()
    df_screamer['screamer'] = df['text'].apply(lambda x: 1 if '!!' in x else 0)
    return df_screamer


def extract_meaningful_words_existence(df):
    tf_idf_difference = helpers.get_meaningful_words_tf_idf_difference(df)
    top_words = tf_idf_difference.iloc[:, 0:20]
    df_abusive_words = pd.DataFrame(columns=['id'] + list(top_words.columns.values))
    df_abusive_words['id'] = df['id'].tolist()
    for word in list(top_words.columns.values):
        df_abusive_words[word] = df['text'].apply(lambda x: 1 if word in x else 0)
    return df_abusive_words


def extract_distance_from_offensive(df):
    tf_idf_difference = helpers.get_meaningful_words_tf_idf_difference(df)
    offensive = tf_idf_difference.iloc[:, 0:100]
    offensive_sentence = ' '.join(list(offensive.columns.values))
    return helpers.get_distance_df(df, 'off_dis', offensive_sentence)


def extract_distance_from_not_offensive(df):
    tf_idf_difference = helpers.get_meaningful_words_tf_idf_difference(df)
    not_offensive = tf_idf_difference.iloc[:, -100:-1]
    not_offensive_sentence = ' '.join(list(not_offensive.columns.values))

    return helpers.get_distance_df(df, 'not_off_dis', not_offensive_sentence)


def extract_distance_from_avg_vector(df):
    neg_posts = utils.get_abusive_df(df)['text'].tolist()
    pos_posts = utils.get_no_abusive_df(df)['text'].tolist()
    m_wiki = w2v.get_model(SOURCE + "/Embedding/wiki.he.word2vec.model")
    m_our = w2v.get_model(SOURCE + "/Embedding/our.corpus.word2vec.model")
    neg_matrix = helpers.create_vectors_array(neg_posts, m_our, m_wiki)
    pos_matrix = helpers.create_vectors_array(pos_posts, m_our, m_wiki)
    neg_avg_vec = np.mean(neg_matrix)
    pos_avg_vec = np.mean(pos_matrix)
    distance_type='euclidean'
    df_offensive_distance = pd.DataFrame(columns=['id', 'dist_avg_neg', 'dist_avg_pos'])
    df_offensive_distance['id'] = df['id'].tolist()
    df_offensive_distance['dist_avg_neg'] = df['text'].apply(
        lambda x:
        utils.calculate_distance(w2v.get_post_vector(m_our, m_wiki, x),
                                 neg_avg_vec, distance_type)
    )
    df_offensive_distance['dist_avg_pos'] = df['text'].apply(
        lambda x:
        utils.calculate_distance(w2v.get_post_vector(m_our, m_wiki, x),
                                 pos_avg_vec, distance_type)
    )
    return df_offensive_distance


def extract_features(df, features):
    functions_dict = get_functions_dictionary()
    features_df = pd.DataFrame(columns=['id'])
    features_df['id'] = df['id'].tolist()
    for feature in features:
        features_df = pd.merge(features_df, functions_dict[feature](df), on='id')
    return features_df
