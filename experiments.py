from sklearn.model_selection import train_test_split
import utils
import preprocessing
import evaluation
import visualization
from feature_extractions import network_feature_extraction, nlp_feature_extractions, user_feature_extraction
import graph
import numpy as np
import XGBoost
import pathlib
import os
import shutil
import networkx as nx
import matplotlib.pyplot as plt

SOURCE = pathlib.Path(__file__).parent

# prepare the data and folders
posts_data = utils.read_to_df()
users_data = utils.concat_posts_per_user(posts_data)
path_object = pathlib.Path(SOURCE / 'outputs')
if path_object.exists():
    shutil.rmtree(SOURCE / 'outputs')
os.makedirs(SOURCE / 'outputs')

# pre process data
print("pre processing data...")
posts_data = preprocessing.preprocess_text(posts_data)
users_data = preprocessing.preprocess_text(users_data)

# create network with topics
print("create network")
topics = graph.get_topics(users_data, 0.09, 5)
network_file_name = SOURCE / 'outputs/bullies_network.csv'
graph.create_csv_network_from_topics(network_file_name, topics)
network_graph = graph.create_graph(network_file_name)
# nx.draw(network_graph, with_labels=True)
# print(nx.info(network_graph))


# pre process network
print("pre processing network...")
network_graph = preprocessing.preprocess_graph(network_graph, 0.2)  # TODO determined threshold

# extract nlp features
print("extract features...")
feature_list = ['post_length', 'tfidf', 'topics', 'screamer', 'words', 'off_dis', 'not_off_dis']
X_nlp = nlp_feature_extractions.extract_features(users_data, feature_list)
y_nlp = (users_data['cb_level'] == 3).astype(int)
X_users = user_feature_extraction.extract_number_of_posts(posts_data)
X_nlp = X_nlp.merge(X_users, on='writer')
X_nlp = X_nlp.drop(columns=['writer'])

# extract network features TODO


# create train set and test set
X_nlp_train, X_nlp_test, y_nlp_train, y_nlp_test = train_test_split(X_nlp, y_nlp, test_size=0.2)

# train the nlp model
print("train models...")
xgb_obj = XGBoost.XGBoost()
xgb_classifier = xgb_obj.train(X_nlp_train, y_nlp_train)
y_nlp_prob_xgb = xgb_obj.predict(X_nlp_test)
y_nlp_pred_xgb = np.where(y_nlp_prob_xgb > 0.5, 1, 0)
performances_nlp = evaluation.get_performances(y_nlp_test, y_nlp_pred_xgb)

# train the network model TODO

# train nlp combined with network model TODO

# evaluation for nlp
print("evaluation for nlp only")
roc_auc_nlp, fpr_nlp, tpr_nlp = evaluation.get_roc_auc(y_nlp_test, y_nlp_prob_xgb)
visualization.plot_roc_curve(roc_auc_nlp, fpr_nlp, tpr_nlp, 'nlp ROC/AUC')

# evaluation for network TODO
print("evaluation for network only")

# evaluation for nlp combined with network TODO
print("evaluation for nlp and network combined")

# comparison for all three
visualization.plot_models_compare_performance(performances_nlp, performances_nlp, performances_nlp)  # TODO: change the second and third 'performances_nlp'
