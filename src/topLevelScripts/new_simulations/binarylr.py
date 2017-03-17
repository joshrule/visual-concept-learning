import time
start_time = time.time()

import os
caffe_home = os.environ['CAFFE_HOME']

import sys
sys.path.insert(0, os.path.join(caffe_home, 'python'))

import caffe
import h5py
import numpy as np
import scipy.io
import shutil
import fnmatch
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score

dirname = sys.argv[1]

# wrangle the data

datafile = os.path.join(dirname, 'data.mat')
mat = scipy.io.loadmat(datafile)

X_tr = mat['X_tr']
y_tr = np.ravel(mat['y_tr'])
w_tr = np.ravel(mat['w_tr'])

X_te = mat['X_te']
y_te = np.ravel(mat['y_te'])

### X,y = make_classification(n_samples=50000, n_features=20, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=None)
### 
### X_tr,X_te,y_tr,y_te = train_test_split(X,y,stratify=y,train_size=0.5)
### w_tr = np.ones(y_tr.shape)

# define the model
lr = LogisticRegression(penalty='l2', tol=1e-4, C=5e-2,
    fit_intercept=True, intercept_scaling=1, random_state=1,
    solver='liblinear', max_iter=100, verbose=3).fit(X_tr, y_tr, sample_weight=w_tr)

# can also be used with L1 or Elastic Net regularization
# lr = SGDClassifier(loss='log', penalty='elasticnet', alpha=5e-2, fit_intercept=True,
#     n_iter=100, shuffle=True, verbose=2, random_state=1, l1_ratio=0.15,
#     learning_rate='optimal', average=False).fit(X_tr, y_tr, sample_weight=w_tr)

# Test the model
features = lr.predict_proba(X_te)
y_hat = lr.predict(X_te)
score = np.mean([x==y for x,y in zip(y_hat,y_te)])
precision,recall,F,support = precision_recall_fscore_support(y_te,y_hat)
# http://stats.stackexchange.com/questions/7207/
y_tmp = y_te
y_tmp.shape = (len(y_te),1)
y_te_all = np.c_[1-y_tmp,y_tmp]
pr_auc_all = average_precision_score(y_te_all,features,average=None)
roc_auc_all = roc_auc_score(y_te_all,features,average=None)
pr_auc = average_precision_score(y_te,np.ravel(features[:,1]),average='micro')
roc_auc = roc_auc_score(y_te,np.ravel(features[:,1]),average='micro')

# Report the results
result_file = os.path.join(dirname, 'results.mat')

scipy.io.savemat(result_file,{'features' : features,
                              'y_hat' : y_hat,
                              'precision' : precision,
                              'recall' : recall,
                              'support' : support,
                              'accuracy' : score,
                              'pr_auc' : pr_auc,
                              'roc_auc' : roc_auc,
                              'F' : F})

stop_time = time.time()
print 'accuracy = {:.3f}'.format(score)
print 'precision: {}'.format(precision)
print 'recall: {}'.format(recall)
print 'F: {}'.format(F)
print 'PR AUC: {}'.format(pr_auc)
print 'ROC AUC: {}'.format(roc_auc)
print 'PR AUC (all): {}'.format(pr_auc_all)
print 'ROC AUC (all): {}'.format(roc_auc_all)
print 'support: {}'.format(support)
print 'time to train and test classifier: {:.2f}s'.format(stop_time-start_time)
