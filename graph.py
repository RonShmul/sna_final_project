import featureExtraction as fe
import csv

def concat_posts_per_user(df):
    """

    :param df:
    :return:
    """
    df = df.groupby('writer').agg({'text':' '.join})
    return df

def get_topics(df, threshold, num_of_topics):
    """

    :param df:
    :param threshold:
    :param num_of_topics:
    :return:
    """
    topics = {}
    df_topics = fe.extract_topics(df, num_of_topics)
    df_topics = df_topics[df_topics[df_topics.columns]>threshold]
    for row in df_topics.iterrows():
        row = row.dropna(axis=1)
        columns = row.columns
        for col in columns:
            topics[col] = topics.setdefault(col, []).append((row['writer'], row[col]))
    return topics

def create_csv_network(topics_dict):
    with open('bullies_network.csv', 'w') as file:
        writer = csv.writer(file)
        for topic, writers_list in topics_dict.items():
            for i, writer_1 in enumerate(writers_list):
                for j in range(i+1,len(writers_list)):
                    writer_2 = writers_list[j]
                    writer.writerow([writer_1[0], writer_2[0],writer_1[1]-writer_2[1]])



