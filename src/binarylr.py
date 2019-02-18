import os
import numpy as np
import scipy.io
from sklearn import svm
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score
import utils

def binary_log_regression(X_tr, y_tr, w_tr, X_te, y_te, out_dir=None, small=True):
    result_file = os.path.join(out_dir, 'results.mat') if out_dir is not None else None

    if result_file is not None and os.path.exists(result_file):
        mat = scipy.io.loadmat(result_file)
        return {'features' : np.ravel(mat['features']),
                'precision' : float(np.ravel(mat['precision'])),
                'recall' : float(np.ravel(mat['recall'])),
                'accuracy' : float(np.ravel(mat['score'])),
                'pr_auc' : float(np.ravel(mat['pr_auc'])),
                'roc_auc' : float(np.ravel(mat['roc_auc'])),
                'F' : float(np.ravel(mat['F'])),
                'dprime' : float(np.ravel(mat['dprime']))}

    # Linear SVM
    lr = svm.SVC(C=0.1, kernel='linear', probability=True, random_state=1,
        tol=1.e-3, max_iter=1000, cache_size=500).fit(X_tr, y_tr, sample_weight=w_tr)

    # Test the model
    features = lr.predict_proba(X_te)
    y_hat = lr.predict(X_te)
    score = np.mean([x==y for x,y in zip(y_hat,y_te)])
    precision,recall,F,support = precision_recall_fscore_support(y_te,y_hat,average='binary')
    pr_auc = average_precision_score(y_te,np.ravel(features[:,1]),average='macro')
    roc_auc = roc_auc_score(y_te,np.ravel(features[:,1]),average='macro')
    dprime = utils.dprime_score(y_hat,y_te)

    # Report the results
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

    if result_file is not None:
        scipy.io.savemat(result_file, results)
    
    return results
