from sklearn.metrics import f1_score, roc_curve, auc
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt


def get_performances(true, pred):
    """
    get dictionary of evaluation of precision, recall and f-measure
    :param true:
    :param pred:
    :return:
    """
    return {
        'f-score': f_measure(true, pred),
        'precision': precision(true, pred),
        'recall': recall(true, pred),
    }


def f_measure(true, pred):
    return f1_score(true, pred, average='binary')


def precision(true, pred):
    return precision_score(true, pred, average='binary')


def recall(true, pred):
    return recall_score(true, pred, average='binary')


def get_roc_auc(y, y_pred):
    fpr, tpr, _ = roc_curve(y, y_pred)
    roc_auc = auc(fpr, tpr)
    return roc_auc, fpr, tpr


def precision_recall(y, y_pred):
    p, r, _ = precision_recall_curve(y, y_pred)
    plt.step(r, p, color='b', alpha=0.2, where='post')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve')
    plt.legend(loc="lower right")


def evaluate_with_cross_validation(model, X, y):
    """
    print evaluations of precision, recall, ROC/AUC and accuracy of a given model with a given train set and target
    return the resukt of the cross validation
    :param model:
    :param X:
    :param y:
    :return:
    """
    cv_results = cross_validate(model, X, y, cv=3,
                                scoring=('accuracy', 'precision', 'recall', 'roc_auc'),
                                return_train_score=True)
    msg = "train_accuracy: %f. test_accuracy: %f)" % (np.mean(cv_results['train_accuracy']),
                                                          np.mean(cv_results['test_accuracy']))
    print(msg)
    msg = "train_precision: %f. test_precision: %f)" % (np.mean(cv_results['train_precision']),
                                                            np.mean(cv_results['test_precision']))
    print(msg)
    msg = "train_recall: %f. test_recall: %f)" % (np.mean(cv_results['train_recall']),
                                                      np.mean(cv_results['test_recall']))
    print(msg)
    msg = "train_roc_auc: %f. test_roc_auc: %f)" % (np.mean(cv_results['train_roc_auc']),
                                                        np.mean(cv_results['test_roc_auc']))
    print(msg)
    return cv_results
