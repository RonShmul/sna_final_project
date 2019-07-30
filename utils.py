import pandas as pd
from nltk.tokenize import word_tokenize
from scipy.spatial import distance
from sklearn.externals import joblib
import pathlib
import matplotlib.pyplot as plt
import base64
SOURCE = pathlib.Path(__file__).parent


def get_abusive_df(df):
    return df.loc[df['cb_level'] == 3]


def get_no_abusive_df(df):
    return df.loc[df['cb_level'] == 1]


def get_tagged_posts(df):
    return df.loc[(df['cb_level'] == 1) | (df['cb_level'] == 3)]


def concat_posts_per_user(df):
    """

    :param df:
    :return:
    """
    df = df.groupby('writer', as_index=False).agg({
        'text': ' '.join,
        'cb_level': lambda cb_levels: cb_levels.value_counts().idxmax()
    })
    return df


def file_to_list(path, encoding='cp1255'):
    """
    Insert file data into list
    :param path:
    :param encoding:
    :return:
    """
    try:
        with open(path, mode='r', encoding=encoding) as infile:
            myList = [line.strip('\n') for line in infile]
    except UnicodeDecodeError as e:
        with open(path,mode='r', encoding='utf-8') as infile:
            myList = [line.strip('\n') for line in infile]
    return myList


def traverse(word):
    """
    Traverse words from left to right
    :param word:
    :return word:
    """
    if type(word) is list:
        return [''.join(wrd[-1:-(len(wrd)+1):-1]) if type(wrd) is str and len(wrd)>0 and wrd[0] in 'אבגדהוזחטיכלמנסעפצקרשת' else wrd for wrd in word]
    elif type(word) is str: return traverse([word])[0]
    elif type(word) is set: return set(traverse(list(word)))
    elif type(word) is dict: dict(zip(traverse(word.keys()), traverse(word.values())))
    elif type(word) == type(pd.Series()): return pd.Series(data=traverse(list(word)), index=word.index, name=word.name)
    elif type(word) == type(type(pd.DataFrame())): return word.applymap(lambda x: traverse(x))
    return word


def get_stop_words():
    """
    gets list of stop words from a file
    :return:
    """
    stop_words = file_to_list(SOURCE / 'stop_words.txt')
    return stop_words


def create_stop_words_list(dataframe, threshold):
    """
    create list of frequent words according to a given threshold
    :param dataframe: dataframe
    :param threshold: double
    :return stop_words: list
    """
    stop_words = get_stop_words()
    text = dataframe.text.tolist()
    term_df = {}
    number_posts = len(text)

    for index_post in range(1, number_posts):
        tokens = word_tokenize(text[index_post])
        for token in tokens:
            if token in term_df:
                list_posts = term_df[token]
                if index_post not in list_posts:
                    term_df[token].append(index_post)  # change from indexes to counter
            else:
                term_df[token] = []
                term_df[token].append(index_post)

    for token, posts in term_df.items():
        df = len(posts)
        df_normal = float(df / number_posts)
        if df_normal > threshold:
            stop_words.append(token)

    return stop_words


def calculate_distance(a, b, distance_type='euclidean'):
    if distance_type == 'euclidean':
        return distance.euclidean(a, b)
    if distance_type == 'cosine':
        return distance.cosine(a, b)
    return None


def read_to_df(path='fixedData.csv'):
    """
    reads the csv data file to a data frame and gets only the tagged post
    :return df:
    """
    try:
        df = pd.read_csv(path)
        if df.shape[0] == 0:
            raise ValueError
        return get_tagged_posts(df)
    except KeyError:
        raise ValueError


def save_model(model, path='outputs/model.pkl'):
    joblib.dump(model, path)


def get_model(path='outputs/model.pkl'):
    path_object = pathlib.Path(path)
    if path_object.exists():
        return joblib.load(path)
    return None


def save_picture(path):
    plt.savefig(path)


def clear_plot():
    plt.clf()


def get_image_string(path):
    with open(path, mode='rb') as file:
        img = file.read()
    return base64.encodebytes(img).decode("utf-8")

