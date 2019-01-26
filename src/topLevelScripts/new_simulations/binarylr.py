import os
import numpy as np
import scipy.io
import shutil
import scipy.stats
from sklearn import svm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
import time

def binary_log_regression(X_tr, y_tr, w_tr, X_te, y_te, out_dir):
    result_file = os.path.join(out_dir, 'results.mat')

    if os.path.exists(result_file):
        mat = scipy.io.loadmat(result_file)
        return (float(mat['precision']), float(mat['recall']), float(mat['accuracy']), 
            float(mat['pr_auc']), float(mat['roc_auc']), float(mat['F']), float(mat['dprime']))

    # Logistic Regression?
    # lr = LogisticRegression(penalty='l2', tol=1e-4, C=5e-2,
    #     fit_intercept=True, intercept_scaling=1, random_state=1,
    #     solver='liblinear', max_iter=100, verbose=0).fit(X_tr, y_tr, sample_weight=w_tr)

    # Cross-Validated Logistic Regression?
    # lr = LogisticRegressionCV(penalty='l2', tol=1e-4, scoring='f1_micro',
    #     fit_intercept=True, random_state=1, solver='saga', max_iter=150,
    #     verbose=0).fit(X_tr, y_tr, sample_weight=w_tr)

    # SGD Logistic Regression?
    # lr = SGDClassifier(loss='log', penalty='elasticnet', alpha=5e-2, fit_intercept=True,
    #     n_iter=100, shuffle=True, verbose=2, random_state=1, l1_ratio=0.15,
    #     learning_rate='optimal', average=False).fit(X_tr, y_tr, sample_weight=w_tr)

    # Linear SVM?
    lr = svm.SVC(C=0.1, kernel='linear', probability=True, random_state=1,
        tol=1.e-3, max_iter=1000, cache_size=500).fit(X_tr, y_tr, sample_weight=w_tr)

    # Test the model
    features = lr.predict_proba(X_te)
    y_hat = lr.predict(X_te)
    score = np.mean([x==y for x,y in zip(y_hat,y_te)])
    precision,recall,F,support = precision_recall_fscore_support(y_te,y_hat,average='binary')
    pr_auc = average_precision_score(y_te,np.ravel(features[:,1]),average='macro')
    roc_auc = roc_auc_score(y_te,np.ravel(features[:,1]),average='macro')
    dprime = dprime_score(y_hat,y_te)

    # Report the results
    scipy.io.savemat(result_file,{'features' : features,
                                  # 'y_hat' : y_hat,
                                  'precision' : float(np.ravel(precision)),
                                  'recall' : float(np.ravel(recall)),
                                  'accuracy' : float(np.ravel(score)),
                                  'pr_auc' : float(np.ravel(pr_auc)),
                                  'roc_auc' : float(np.ravel(roc_auc)),
                                  'F' : float(np.ravel(F)),
                                  'dprime' : float(np.ravel(dprime))})
    
    return (precision, recall, score, pr_auc, roc_auc, F, dprime)

def dprime_score(y_hats,ys):
    truepositives = sum(h == 1 and y == 1 for h, y in zip(y_hats, ys))
    falsepositives = sum(h == 1 and y == 0 for h, y in zip(y_hats, ys))
    truenegatives = sum(h == 0 and y == 0 for h, y in zip(y_hats, ys))
    falsenegatives = sum(h == 0 and y == 1 for h, y in zip(y_hats, ys))
    numpositive = truepositives + falsenegatives
    numnegative = falsepositives + truenegatives
    
    tpr = float(truepositives)/float(numpositive)
    fpr = float(falsepositives)/float(numnegative)
    
    if (tpr == 1.0):
        # conventional correction to avoid infinite D'
        tpr = float(numpositive + numnegative - 1) / float(numpositive + numnegative)
    elif (tpr == 0.0):
        # conventional correction to avoid -infinite D'
        tpr = 1.0 / float(numpositive + numnegative)
    if (fpr == 1.0):
        # conventional correction to avoid infinite D'
        fpr = float(numpositive + numnegative - 1) / float(numpositive + numnegative)
    elif (fpr == 0.0):
        # conventional correction to avoid -infinite D'
        fpr = 1.0 / float(numpositive + numnegative)
    return scipy.stats.norm.ppf(tpr) - scipy.stats.norm.ppf(fpr)
