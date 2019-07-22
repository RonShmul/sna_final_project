import utils
import pandas as pd
from feature_extractions import nlp_feature_extractions
import numpy as np


def num_of_posts_per_column(df, column_name="source"):
    """
    number of abusive posts for a given column
    :param df:
    :param column_name:
    :return:
    """
    # columns: column_name, total_count, total_from_corpus, number_of_abusive, normalized_abusive
    total_count_df = df.groupby(column_name)['cb_level'].apply(lambda x: x.count())
    total_from_corpus_df = df.groupby(column_name)['cb_level'].apply(lambda x: (x.count() / df.shape[0]))
    number_of_abusive_df = df.groupby(column_name)['cb_level'].apply(lambda x: x[x == 3].count())
    number_of_not_abusive_df = df.groupby(column_name)['cb_level'].apply(lambda x: x[x == 1].count())
    normalized_abusive_df = df.groupby(column_name)['cb_level'].apply(lambda x: (x[x == 3].count() / x.count()) * 100)
    normalized_not_abusive_df = df.groupby(column_name)['cb_level'].apply(
        lambda x: (x[x == 1].count() / x.count()) * 100)

    result = pd.DataFrame({'total_count': total_count_df}) \
        .merge(pd.DataFrame({'total_from_corpus': total_from_corpus_df}), on=[column_name], right_index=True)
    result = result \
        .merge(pd.DataFrame({'number_of_abusive': number_of_abusive_df}), on=[column_name], right_index=True)
    result = result \
        .merge(pd.DataFrame({'normalized_abusive': normalized_abusive_df}), on=[column_name], right_index=True)
    result = result \
        .merge(pd.DataFrame({'number_of_not_abusive': number_of_not_abusive_df}), on=[column_name], right_index=True)
    result = result \
        .merge(pd.DataFrame({'normalized_not_abusive': normalized_not_abusive_df}), on=[column_name], right_index=True)

    return result


def avg_per_class(df):
    df_abusive = utils.get_abusive_df(df)
    df_no_abusive = utils.get_no_abusive_df(df)
    length_abusive_df = nlp_feature_extractions.extract_post_length(df_abusive)
    avg_length_abusive = np.mean(length_abusive_df['post_length'])
    length_no_abusive_df = nlp_feature_extractions.extract_post_length(df_no_abusive)
    avg_length_no_abusive = np.mean(length_no_abusive_df['post_length'])
    dictionary_length = {'abusive': avg_length_abusive,
                         'no abusive': avg_length_no_abusive}
    return dictionary_length


def posts_per_user(df):
    return df.groupby('writer', as_index=False).agg({'text': lambda posts: posts.shape[0]})


def users_per_source(df):
    return df.groupby('source', as_index=False).agg({'writer': lambda writers: writers.shape[0]})

####################################################################################################################
# run statistics
####################################################################################################################


# prepare the data
tagged_df = utils.read_to_df()
users_df = utils.concat_posts_per_user(tagged_df)
num_of_posts_per_column_df = num_of_posts_per_column(tagged_df)
posts_per_user_df = posts_per_user(tagged_df)
users_per_source_df = users_per_source(tagged_df)
print('done')