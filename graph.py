from feature_extractions import nlp_feature_extractions as fe
import csv
import networkx as nx
import pandas as pd


def get_topics(df, threshold, num_of_topics):
    """

    :param df:
    :param threshold:
    :param num_of_topics:
    :return:
    """
    topics = {}
    df_topics = fe.extract_topics(df, num_of_topics)
    df_topics = df_topics[df_topics[df_topics.columns] > threshold]
    for i, row in df_topics.iterrows():
        row = row.dropna()
        columns = row.index.tolist()
        columns.remove('writer')
        for col in columns:
            topics.setdefault(col, []).append((row['writer'], row[col]))
    return topics


def create_csv_network(file_name, topics_dict):
    """

    :param topics_dict:
    :return:
    """
    with open(file_name, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(['Node A', 'Node B', 'Weight'])
        for topic, writers_list in topics_dict.items():
            for i, writer_1 in enumerate(writers_list):
                for j in range(i+1, len(writers_list)):
                    writer_2 = writers_list[j]
                    writer.writerow([writer_1[0], writer_2[0], writer_1[1]-writer_2[1]])


def create_graph(csv_file):
    """

    :param csv_file:
    :return:
    """
    graph_edges = pd.read_csv(csv_file)
    graph_temp = nx.from_pandas_edgelist(graph_edges, source='Node A', target='Node B', edge_attr='Weight')
    graph = nx.to_undirected(graph_temp)
    graph_new = nx.Graph(graph)
    return graph_new
