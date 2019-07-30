import re
import networkx as nx


def preprocess_text(df):
    """
    get a dataframe - keepersData and preprocess its data and return the result dataframe
    :param df: dataframe
    :return clean_df: dataframe
    """
    if 'text' not in df.columns and 'writer' not in df.columns or df.shape[0] == 0:
        raise ValueError
    df.dropna(inplace=True)
    df['text'] = df['text'].apply(
        lambda x:
        re.sub(r'[^א-ת!]+', ' ', x)
    )
    df.reset_index(drop=True, inplace=True)
    return df


def preprocess_graph(graph, threshold):
    """

    :param graph:
    :param threshold:
    :return:
    """
    remove = [edge for edge in graph.edges().items() if edge[1]['Weight'] > threshold]
    remove_list = [remove[i][0] for i in range(len(remove))]
    graph.remove_edges_from(remove_list)
    isolated = list(nx.isolates(graph))  # isolate and remove the unconnected nodes
    graph.remove_nodes_from(isolated)
    return graph
