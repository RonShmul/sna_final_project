import featureExtraction as fe
import csv
import networkx as nx
import pandas as pd


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
    """

    :param topics_dict:
    :return:
    """
    with open('bullies_network.csv', 'w') as file:
        writer = csv.writer(file)
        for topic, writers_list in topics_dict.items():
            for i, writer_1 in enumerate(writers_list):
                for j in range(i+1,len(writers_list)):
                    writer_2 = writers_list[j]
                    writer.writerow([writer_1[0], writer_2[0],writer_1[1]-writer_2[1]])


def create_graph(csv_file):
    """

    :param csv_file:
    :return:
    """
    graph_edges = pd.read_csv(csv_file, columns=['Node A', 'Node B', 'Weight'])
    graph_temp = nx.from_pandas_edgelist(graph_edges, source='Node A', target='Node B', edge_attr='Weight')
    graph = nx.to_undirected(graph_temp)
    graph_new = nx.Graph(graph)
    return graph_new


def preprocess(graph, threshold):
    """

    :param graph:
    :param threshold:
    :return:
    """
    remove = [edge for edge in graph.edges().items() if edge[1]['Weight'] < threshold]
    remove_list = [remove[i][0] for i in range(len(remove))]
    graph.remove_edges_from(remove_list)
    isolated = list(nx.isolates(graph))  # isolate and remove the unconnected nodes
    graph.remove_nodes_from(isolated)
    return graph