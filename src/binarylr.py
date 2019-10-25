import gzip
import numpy as np
import os
import pickle
from sklearn import svm
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
import utils

def binary_log_regression(X_tr, y_tr, w_tr, X_te, y_te, out_dir=None, small=True):
    """ a wrapper around binary classification

        I wrap the classifier to look for cached results, save any newly
        computed results, and compute additional summary statistics about
        classifier performance.

        The name is a poor reflection of the purpose. It's called
        `binary_log_regression, but it's isn't necessarily logistic regression,
        but just some sort of binary classifier. I use logistic regression now,
        but it was a Linear SVM for a while.
    """
    # Save the results if provided with information about where to save them.
    result_file = os.path.join(out_dir, 'results.pkl') if out_dir is not None else None

    # Return cached results when available.
    if result_file is not None and os.path.exists(result_file):
        with gzip.open(result_file, 'rb') as fd:
            result = pickle.load(fd)
        return result

    # Classify using a Linear SVM.
    # lr = svm.SVC(C=0.1, kernel='linear', probability=True, random_state=1,
    #     tol=1.e-3, max_iter=1000, cache_size=500).fit(X_tr, y_tr, sample_weight=w_tr)

    # Classify using Logistic Regression. This is a bit complex since it uses
    # the *CV method and so needs to make sure enough data is available to
    # create splits.
    n_folds = int(min(3, X_tr.shape[0]/2))
    if n_folds <= 1:
        lr = LogisticRegression(fit_intercept=True, dual=False,
            penalty='elasticnet', solver='saga', tol=0.0001, max_iter=200,
            random_state=1, C=1.0, l1_ratio=0.5).fit(X_tr, y_tr, sample_weight=w_tr)
    else:
        lr = LogisticRegressionCV(fit_intercept=True, dual=False,
            penalty='elasticnet', solver='saga', tol=0.0001, max_iter=200,
            refit=True, random_state=1, cv=n_folds, Cs=10,
            l1_ratios=[0.0, 0.25, 0.5, 0.75, 1.0]).fit(X_tr, y_tr, sample_weight=w_tr)

    # Test the model.
    features = lr.predict_proba(X_te)
    y_hat = lr.predict(X_te)
    score = np.mean([x==y for x,y in zip(y_hat,y_te)])
    precision,recall,F,support = precision_recall_fscore_support(y_te,y_hat,average='binary')
    pr_auc = average_precision_score(y_te,np.ravel(features[:,1]),average='macro')
    roc_auc = roc_auc_score(y_te,np.ravel(features[:,1]),average='macro')
    dprime = utils.dprime_score(y_hat,y_te)

    # Report the results.
    results = {'precision' : float(np.ravel(precision)),
               'recall' : float(np.ravel(recall)),
               'accuracy' : float(np.ravel(score)),
               'pr_auc' : float(np.ravel(pr_auc)),
               'roc_auc' : float(np.ravel(roc_auc)),
               'F' : float(np.ravel(F)),
               'dprime' : float(np.ravel(dprime))}
    if not small:
        results['model'] = lr
        results['features'] = features

    # Save the results
    if result_file is not None:
        with gzip.open(result_file, 'wb') as fd:
            pickle.dump(results, fd)
    
    return results
