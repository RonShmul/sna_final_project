from sklearn.model_selection import train_test_split
import utils
import preprocessing
import evaluation
import visualization
from feature_extractions import network_feature_extraction, nlp_feature_extractions
import graph
import numpy as np
import XGBoost
import pathlib
import os
import shutil
SOURCE = pathlib.Path(__file__).parent

# prepare the data and folders
tagged_df = utils.read_to_df()
tagged_df = utils.concat_posts_per_user(tagged_df)
path_object = pathlib.Path(SOURCE / 'outputs')
if path_object.exists():
    shutil.rmtree(SOURCE / 'outputs')
os.makedirs(SOURCE / 'outputs')

# create network TODO
print("create network")


# pre process data
print("pre processing...")
tagged_df = preprocessing.preprocess_text(tagged_df)

# pre process network TODO

# extract nlp features
print("extract feature...")
feature_list = ['post_length', 'tfidf', 'topics', 'screamer', 'words', 'off_dis', 'not_off_dis']
X = nlp_feature_extractions.extract_features(tagged_df, feature_list)
y = (tagged_df['cb_level'] == 3).astype(int)
X = X.drop(columns=['writer'])

# extract network features TODO


# create train set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# train the nlp model
print("train models...")
xgb_obj = XGBoost.XGBoost()
xgb_classifier = xgb_obj.train(X_train, y_train)
y_prob_xgb = xgb_obj.predict(X_test)
y_pred_xgb = np.where(y_prob_xgb > 0.5, 1, 0)
performances_nlp = evaluation.get_performances(y_test, y_pred_xgb)

# train the network model TODO

# train nlp combined with network model TODO

# evaluation for nlp
print("evaluation for nlp only")
roc_auc_nlp, fpr_nlp, tpr_nlp = evaluation.get_roc_auc(y_test, y_prob_xgb)
visualization.plot_roc_curve(roc_auc_nlp, fpr_nlp, tpr_nlp, 'nlp ROC/AUC')

# evaluation for network TODO
print("evaluation for network only")

# evaluation for nlp combined with network TODO
print("evaluation for nlp and network combined")

# comparison for all three
visualization.plot_models_compare_performance(performances_nlp, performances_nlp, performances_nlp)  # TODO: change the second and third 'performances_nlp'
