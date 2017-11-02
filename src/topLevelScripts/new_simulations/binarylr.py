import os
import numpy as np
import scipy.io
import shutil
from sklearn import svm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
import time

def binary_log_regression(X_tr, y_tr, w_tr, X_te, y_te, out_dir):
    result_file = os.path.join(out_dir, 'results.mat')
    # print 'result_file', result_file, time.time()

    if os.path.exists(result_file):
        mat = scipy.io.loadmat(result_file, variable_names=['precision',
            'recall', 'accuracy', 'pr_auc', 'roc_auc', 'F'])
        return (int(mat['precision']), int(mat['recall']), int(mat['accuracy']), 
                int(mat['pr_auc']), int(mat['roc_auc']), int(mat['F']))
    
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

    # Report the results
    scipy.io.savemat(result_file,{'features' : features,
                                  'y_hat' : y_hat,
                                  'precision' : int(np.ravel(precision)),
                                  'recall' : int(np.ravel(recall)),
                                  'accuracy' : int(np.ravel(score)),
                                  'pr_auc' : int(np.ravel(pr_auc)),
                                  'roc_auc' : int(np.ravel(roc_auc)),
                                  'F' : int(np.ravel(F))})

    return (precision, recall, score, pr_auc, roc_auc, F)
