import pandas as pd
import networkx as nx

def get_functions_dictionary():
    """
    return function dictionary with all the feature extractions functions
    :return:
    """
    return {
        'clustering_coefficient': extract_clustering_coefficient,
        'degree': extract_degree_centrality,
        'betweenness': extract_betweenness_centrality,
        'closeness': extract_closeness_centrality,
        'eigenvector': extract_eigenvector_centrality
    }


degree_dict = {}
closeness_dict = {}
betweenness_dict = {}


# extract clustering coefficient feature
def extract_clustering_coefficient(df, graph):
    df_cc = pd.DataFrame(columns=['writer', 'clustering_coefficient'])
    df_cc['writer'] = df['writer'].tolist()
    cc_dict = nx.clustering(graph)
    df_cc['clustering_coefficient'] = df['writer'].map(cc_dict)
    return df_cc

# extract degree centrality feature
def extract_degree_centrality(df, graph):
    global degree_dict
    df_degree = pd.DataFrame(columns=['writer', 'degree'])
    df_degree['writer'] = df['writer'].tolist()
    degree_dict = nx.degree_centrality(graph)
    df_degree['degree'] = df_degree['writer'].map(degree_dict)
    return df_degree

# extract betweenness centrality feature
def extract_betweenness_centrality(df, graph):
    global betweenness_dict
    df_bet = pd.DataFrame(columns=['writer', 'betweenness'])
    df_bet['writer'] = df['writer'].tolist()
    betweenness_dict = nx.betweenness_centrality(graph)
    df_bet['betweenness'] = df_bet['writer'].map(betweenness_dict)
    return df_bet

# extract closeness centrality feature
def extract_closeness_centrality(df, graph):
    global closeness_dict
    df_closeness = pd.DataFrame(columns=['writer', 'closeness'])
    df_closeness['writer'] = df['writer'].tolist()
    closeness_dict = nx.closeness_centrality(graph)
    df_closeness['closeness'] = df_closeness['writer'].map(closeness_dict)
    return df_closeness

# extract eigenvector centrality feature
def extract_eigenvector_centrality(df, graph):
    df_eigen = pd.DataFrame(columns=['writer', 'eigenvector'])
    df_eigen['writer'] = df['writer'].tolist()
    eigen_dict = nx.eigenvector_centrality(graph)
    df_eigen['eigenvector'] = df_eigen['writer'].map(eigen_dict)
    return df_eigen


def extract_features(df, features, graph):
    """
    main function returns a data frame with all the given feature extracted from a given data frame
    :param graph:
    :param df:
    :param features:
    :return:
    """
    functions_dict = get_functions_dictionary()
    features_df = pd.DataFrame(columns=['writer'])
    features_df['writer'] = df['writer'].tolist()
    for feature in features:
        features_df = pd.merge(features_df, functions_dict[feature](df, graph), on='writer')
    return features_df
