import pandas as pd
import networkx as nx

def get_functions_dictionary():
    """
    return function dictionary with all the feature extractions functions
    :return:
    """
    return {
        'clustering_coefficient': extract_clustering_coefficient

    }


# extract clustering coefficient feature
def extract_clustering_coefficient(df, graph): #todo: change- take writer from graph
    df_cc = pd.DataFrame(columns=['writer', 'clustering_coefficient'])
    df_cc['writer'] = df['writer'].tolist()
    cc = nx.clustering(graph)
    df_cc['clustering_coefficient'] = df['writer'].map(cc)
    return df_cc

# extract degree centrality feature
def extract_degree(df, graph): # todo
    df_degree = pd.DataFrame(columns=['writer', 'degree'])
    df_degree['writer'] = df['writer'].tolist()
    degree_cen = nx.degree_centrality(graph)
    deg = list(graph.degree())

# extract betweenness centrality feature
    # eigen_cen = nx.eigenvector_centrality(tn)
    # closeness_cen = nx.closeness_centrality(tn)
    # between_cen = nx.betweenness_centrality(tn)

def extract_features(df, features):
    """
    main function returns a data frame with all the given feature extracted from a given data frame
    :param df:
    :param features:
    :return:
    """
    functions_dict = get_functions_dictionary()
    features_df = pd.DataFrame(columns=['writer'])
    features_df['writer'] = df['writer'].tolist()
    for feature in features:
        features_df = pd.merge(features_df, functions_dict[feature](df), on='writer')
    return features_df
