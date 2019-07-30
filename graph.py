from feature_extractions import nlp_feature_extractions as fe
from feature_extractions import network_feature_extraction as net
import csv
import networkx as nx
import pandas as pd
import Embedding.word2vec as word2vec
import pathlib
import utils
from networkx.algorithms import community
import matplotlib.pyplot as plt
from random import randint

SOURCE = pathlib.Path(__file__).parent

def graph_attributes(graph):
    """
    prints graph's attributes
    :param graph:
    :return:
    """
    avg_cluster = nx.average_clustering(graph)
    density = nx.density(graph)
    print(nx.info(graph))
    print('average clustering: ', avg_cluster)
    print('density: ', density)


def get_topics(df, threshold, num_of_topics):
    """
    extract topics with LDA and returns a dictionary with topics as keys
    and list of nodes related to the topic
    :param df:
    :param threshold: float
    :param num_of_topics: int
    :return topics: dictionary
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


def create_csv_network_from_topics(file_name, topics_dict):
    """
    create csv file for the network from the topics dictionary
    :param file_name: string
    :param topics_dict: dictionary
    :return:
    """
    with open(file_name, 'w', encoding='utf-8') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Node A', 'Node B', 'Weight'])
        for topic, writers_list in topics_dict.items():
            for i, writer_1 in enumerate(writers_list):
                for j in range(i+1, len(writers_list)):
                    writer_2 = writers_list[j]
                    csv_writer.writerow([writer_1[0], writer_2[0], abs(writer_1[1]-writer_2[1])])


def create_csv_network_with_word2vec(file_name, users_df):
    with open(file_name, 'w') as file:
        csv_writer = csv.writer(file)
        word2vec_our_model = word2vec.get_model(str(SOURCE / 'Embedding/our.corpus.word2vec.model'))
        word2vec_wiki_model = word2vec.get_model(str(SOURCE / 'Embedding/wiki.he.word2vec.model'))
        csv_writer.writerow(['Node A', 'Node B', 'Weight'])
        writers_list = users_df['writer'].tolist()
        for i, writer_1 in enumerate(writers_list):
            for j in range(i+1, len(writers_list)):
                writer_2 = writers_list[j]
                writer_1_vector = word2vec.get_post_vector(word2vec_our_model, word2vec_wiki_model,
                                                           users_df.loc[users_df['writer'] == writer_1]['text'].item())
                writer_2_vector = word2vec.get_post_vector(word2vec_our_model, word2vec_wiki_model,
                                                           users_df.loc[users_df['writer'] == writer_2]['text'].item())
                csv_writer.writerow([writer_1,
                                     writer_2,
                                     utils.calculate_distance(writer_1_vector, writer_2_vector)])


def create_graph(csv_file):
    """
    reads the csv file with the edges and builds a graph with NetworkX
    :param csv_file: string
    :return graph_new: NetworkX graph
    """
    graph_edges = pd.read_csv(csv_file)
    graph_temp = nx.from_pandas_edgelist(graph_edges, source='Node A', target='Node B', edge_attr='Weight')
    graph = nx.to_undirected(graph_temp)
    graph_new = nx.Graph(graph)
    return graph_new


def find_communities(graph):
    """
    finds partition to 2 communities for the graph
    :param graph: NetworkX graph
    :return com:
    """
    communities_generator = community.girvan_newman(graph)
    com = tuple(sorted(c) for c in next(communities_generator))
    comm_dict = dict(enumerate(com))
    print(comm_dict)
    partition = dict()
    for key in comm_dict:
        for item in comm_dict[key]:
            partition[item] = key
    final_df = pd.DataFrame.from_dict(partition, orient='index')
    print('Partition ' + str(271), final_df)
    return com


def show_communities_graph(graph, partition):
    """
    show the partition to communities with matplotlib
    :param graph:
    :param partition:
    :return:
    """
    d = nx.degree_centrality(graph)
    pos = nx.spring_layout(graph)
    colors = ['#c20078', '#8e82fe', '#feb308', '#02c14d']
    for i in range(len(partition)):
        sub = graph.subgraph(partition[i])
        deg_size = [(d[node] * 1000) for node in sub.node]
        nx.draw_networkx_nodes(sub, pos, node_size=deg_size, node_color=colors[i%4])
        nx.draw_networkx_edges(sub, pos, alpha=0.3)
        nx.draw_networkx_labels(sub, pos, font_size=7, font_color='black')
    plt.show()


def top_10_centrality():
    """
    finds top 10 values of each centrality in graph
    :return:
    """
    top_10_degree = dict(sorted(net.degree_dict.items(), key=lambda x: x[1], reverse=True)[:10])
    top_10_closeness = dict(sorted(net.closeness_dict.items(), key=lambda x: x[1], reverse=True)[:10])
    top_10_between = dict(sorted(net.betweenness_dict.items(), key=lambda x: x[1], reverse=True)[:10])
    df_deg = pd.DataFrame.from_dict(top_10_degree, orient='index')
    df_close = pd.DataFrame.from_dict(top_10_closeness, orient='index')
    df_between = pd.DataFrame.from_dict(top_10_between, orient='index')
    print('top ten degree centrality', df_deg)
    print('top ten closeness centrality', df_close)
    print('top ten between centrality', df_between)
    return top_10_between, top_10_closeness, top_10_degree


def correlation(between, closeness, degree):
    """
    returns the corelated nodes from all centrality
    :param between:
    :param closeness:
    :param degree:
    :return:
    """
    between_list = list(between.keys())
    closeness_list = list(closeness.keys())
    degree_list = list(degree.keys())
    print('correlation between centrality measures: ',list(set(between_list) & set(closeness_list) & set(degree_list)))


def check_power_law(graph):
    plt.hist(sorted([d for n, d in graph.degree()], reverse=True))
    plt.title("Degree Histogram")
    plt.xlabel('Degree')
    plt.ylabel('Number of Subjects')
    plt.savefig('network_degree.png')
    plt.show()