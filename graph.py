import featureExtraction as fe


def concat_posts_per_user(df):
    df = df.groupby('writer').agg({'text':' '.join})
    return df

def get_topics(df, threshold, num_of_topics):
    topics = {}
    df_topics = fe.extract_topics(df, num_of_topics)
    df_topics = df_topics[df_topics[df_topics.columns]>threshold]
    for line in iterrow(df_topics):
        if
