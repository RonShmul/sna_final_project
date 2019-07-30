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
import pandas as pd

SOURCE = pathlib.Path(__file__).parent

# prepare the data and folders #
posts_data = utils.read_to_df()
users_data = utils.concat_posts_per_user(posts_data)
path_object = pathlib.Path(SOURCE / 'outputs')
if path_object.exists():
    shutil.rmtree(SOURCE / 'outputs')
os.makedirs(SOURCE / 'outputs')

# pre process data #
print("pre processing data...")
posts_data = preprocessing.preprocess_text(posts_data)
users_data = preprocessing.preprocess_text(users_data)

# create network with topics #
print("create network")
topics = graph.get_topics(users_data, 0.1, 5)
network_file_name = SOURCE / 'outputs/bullies_network.csv'
graph.create_csv_network_from_topics(network_file_name, topics)
network_graph = graph.create_graph(network_file_name)

# # pre process network #
print("pre processing network...")
network_graph = preprocessing.preprocess_graph(network_graph, 0.1) #todo change back to 0.1
graph.graph_attributes(network_graph)

# extract nlp features #
print("extract nlp features...")
feature_list = ['post_length', 'tfidf', 'topics', 'screamer', 'words', 'off_dis', 'not_off_dis']
X_nlp = nlp_feature_extractions.extract_features(users_data, feature_list)
y_nlp = (users_data['cb_level'] == 3).astype(int)
X_users = user_feature_extraction.extract_number_of_posts(posts_data)
X_nlp = X_nlp.merge(X_users, on='writer')

# extract network features #
print("extract network features...")
feature_list = ['clustering_coefficient', 'degree', 'betweenness', 'closeness']
X_network = network_feature_extraction.extract_features(users_data, feature_list, network_graph)
y_network = (users_data['cb_level'] == 3).astype(int)
X_network = X_network.merge(X_users, on='writer')

# combined data frame #
X_combined = pd.merge(X_nlp, X_network, on='writer')
y_combined = (users_data['cb_level'] == 3).astype(int)

# remove writer's column from data frames #
X_nlp = X_nlp.drop(columns=['writer'])
X_network = X_network.drop(columns=['writer'])
X_combined = X_combined.drop(columns=['writer'])

# create train set and test set #
X_nlp_train, X_nlp_test, y_nlp_train, y_nlp_test = train_test_split(X_nlp, y_nlp, test_size=0.2, random_state=42)
X_network_train, X_network_test, y_network_train, y_network_test = train_test_split(X_network, y_network, test_size=0.2, random_state=42)
X_combined_train, X_combined_test, y_combined_train, y_combined_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42)

# train the nlp model #
print("train models...")
xgb_obj = XGBoost.XGBoost()
xgb_classifier = xgb_obj.train(X_nlp_train, y_nlp_train)
y_nlp_prob_xgb = xgb_obj.predict(X_nlp_test)
y_nlp_pred_xgb = np.where(y_nlp_prob_xgb > 0.5, 1, 0)
performances_nlp = evaluation.get_performances(y_nlp_test, y_nlp_pred_xgb)
print('nlp performances:')
for k,v in performances_nlp.items():
    print(k, v)

# train the network model #
xgb_obj = XGBoost.XGBoost()
xgb_classifier_net = xgb_obj.train(X_network_train, y_network_train)
y_network_prob_xgb = xgb_obj.predict(X_network_test)
y_network_pred_xgb = np.where(y_network_prob_xgb > 0.5, 1, 0)
performances_network = evaluation.get_performances(y_network_test, y_network_pred_xgb)
print('network performances:')
for k,v in performances_network.items():
    print(k, v)

# train nlp combined with network model #
xgb_obj = XGBoost.XGBoost()
xgb_classifier_combined = xgb_obj.train(X_combined_train, y_combined_train)
y_combined_prob_xgb = xgb_obj.predict(X_combined_test)
y_combined_pred_xgb = np.where(y_combined_prob_xgb > 0.5, 1, 0)
performances_combined = evaluation.get_performances(y_combined_test, y_combined_pred_xgb)
print('combined performances:')
for k,v in performances_combined.items():
    print(k, v)

# evaluation for nlp #
print("evaluation for nlp only")
roc_auc_nlp, fpr_nlp, tpr_nlp = evaluation.get_roc_auc(y_nlp_test, y_nlp_prob_xgb)
visualization.plot_roc_curve(roc_auc_nlp, fpr_nlp, tpr_nlp, 'nlp ROC/AUC')

# evaluation for network #
print("evaluation for network only")
roc_auc_network, fpr_network, tpr_network = evaluation.get_roc_auc(y_network_test, y_network_prob_xgb)
visualization.plot_roc_curve(roc_auc_network, fpr_network, tpr_network, 'network ROC/AUC')

# evaluation for nlp combined with network #
print("evaluation for nlp and network combined")
roc_auc_combined, fpr_combined, tpr_combined = evaluation.get_roc_auc(y_combined_test, y_combined_prob_xgb)
visualization.plot_roc_curve(roc_auc_combined, fpr_combined, tpr_combined, 'combined ROC/AUC')

# comparison for all three #
visualization.plot_models_compare_performance(performances_nlp, performances_network, performances_combined)


#################### Network analysis ##################

# find communities #
print('find communities...')
communities = graph.find_communities(network_graph)
graph.show_communities_graph(network_graph,communities)

# find top ten from centrality #
print('Top 10...')
top_10_between, top_10_closeness, top_10_degree = graph.top_10_centrality()

# find correlation between centrality
print('find correlation...')
graph.correlation(top_10_between, top_10_closeness, top_10_degree)

# check if the graph is scale free
print('Check degree distribution...')
graph.check_power_law(network_graph)