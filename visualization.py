import matplotlib.pyplot as plt
import os
import pandas as pd
import xgboost as xgb
import wordcloud
import numpy as np
import utils
import pathlib

WHERE_OUTPUTS = pathlib.Path(__file__).parent


def plot_dictionary(dictionary, title):
    df2 = pd.DataFrame.from_dict(dictionary, orient='index').sort_values(by=0, ascending=False)
    pl = df2.plot(kind='bar', figsize=(15, 7), fontsize=8, legend=False, title=utils.traverse(title))
    for p in pl.patches:
        pl.annotate(str(p.get_height()), (p.get_x() * 0.98, p.get_height() * 1.001), fontsize=14)
    plt.show()


def create_word_cloud(no_topics, lda, feature_names):
    for i in range(0, no_topics):
        d = dict(zip(utils.traverse(feature_names), lda.components_[i]))
        wc = wordcloud.WordCloud(background_color='white', max_words=50, stopwords=utils.get_stop_words())
        image = wc.generate_from_frequencies(d)
        image.to_file(WHERE_OUTPUTS / 'outputs' + r'\Topic' + str(i+1) + '.png')
        plt.figure()
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.show()


def create_lda_visualization(no_topics, lda_model):
    tf_vectorizer = utils.get_model(os.path.join(WHERE_OUTPUTS / 'outputs', 'tf.pkl'))
    tf_feature_names = tf_vectorizer.get_feature_names()

    create_word_cloud(no_topics, lda_model, tf_feature_names)


def plot_roc_curve(roc_auc, fpr, tpr, name):
    plt.figure()
    lw = 1
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='-')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic -'+str(name))
    plt.legend(loc="lower right")
    plt.show()


def plot_feature_importance_xgboost(booster):
    xgb.plot_importance(booster, importance_type='gain')
    plt.rcParams['figure.figsize'] = [5, 5]
    plt.show()


def plot_models_compare_performance(nlp_only, network_only, nlp_network):
    fig, ax = plt.subplots()
    index = np.arange(3)
    ax.bar(index, [nlp_only[key] for key in sorted(nlp_only.keys())],
           color=(0.5, 0.4, 0.8, 0.4), width=0.2, label='nlp')
    ax.bar(index+0.33, [network_only[key] for key in sorted(network_only.keys())],
           color=(0.8, 0.5, 0.4, 0.6), width=0.2, label='network')
    ax.bar(index+0.66, [nlp_network[key] for key in sorted(nlp_network.keys())],
           color=(0.2, 0.8, 0.4, 0.6), width=0.2, label='nlp and network')
    ax.set_xlabel('Performances')
    ax.set_ylabel('')
    ax.set_title('Model compare')
    ax.set_xticks(index+0.3)
    ax.set_xticklabels(['F-Measure', 'Precision', 'Recall'])
    ax.legend()
    fig.tight_layout()
    plt.show()

