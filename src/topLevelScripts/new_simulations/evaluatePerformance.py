import binarylr as blr
import gc
import h5py
import joblib
import numpy as np
import os
import pandas
import scipy.io
import shutil
import sklearn.preprocessing as skp
import sys
import tempfile
import time

def evaluatePerformance(data_file, out_file, tmp_dir=None):

    print 'setting up evaluatePerformance:', time.time()
    mat = scipy.io.loadmat(data_file)

    tr_file = str(mat['tr_file'][0])
    te_file = str(mat['te_file'][0])
    score_file = str(mat['score_file'][0])

    eval_dir = str(mat['eval_dir'][0])
    eval_N = int(mat['eval_N'])
    eval_type = float(mat['thresh'])

    n_train = np.ravel(mat['nTrain'])
    n_runs = int(mat['nRuns'])
    n_features = float(mat['nFeatures'])
    classes = np.ravel(mat['classes'])

    n_training_examples = n_train.size
    n_classes = classes.size

    print 'configuring memmaps:', time.time()
    if tmp_dir is None:
        tmp_home = os.path.expanduser('~/.sim_tmp')
        if not os.path.exists(tmp_home):
            os.mkdir(tmp_home)
        tmp_dir = tempfile.mkdtemp(dir=tmp_home)
    X_te = load_var(tmp_dir, 'X_te.mmap', te_file, 'x')
    y_te = load_var(tmp_dir, 'y_te.mmap', te_file, 'y')
    X_tr = load_var(tmp_dir, 'X_tr.mmap', tr_file, 'x')
    y_tr = load_var(tmp_dir, 'y_tr.mmap', tr_file, 'y')
    if score_file == tr_file:
        scores = X_tr
    else:
        scores = load_var(tmp_dir, 'scores.mmap', score_file, 'x')

    # running the classification problems
    print 'running classifications:', time.time()
    # data = joblib.Parallel(n_jobs=-1, verbose=2, max_nbytes=None)(
    data = joblib.Parallel(n_jobs=-1, verbose=10, max_nbytes=1e6, mmap_mode='r')(
            joblib.delayed(run_classifier)(X_tr, y_tr, X_te, y_te, scores, 
                classes, eval_dir, eval_N, eval_type, n_train, n_features,
                i_run, i_train, i_class)
            for i_run in xrange(n_runs) 
            for i_train in xrange(n_training_examples)
            for i_class in xrange(n_classes))
    # save the results to a table
    print 'saving results:', time.time()
    columns = ('class', 'iClass', 'nTraining', 'iTrain', 'iRun', 'out_dir', 
            'eval_n', 'eval_type', 'precision', 'recall', 'support',
            'score', 'pr_auc', 'roc_auc', 'F')
    df = pandas.DataFrame.from_records(data, columns=columns)
    df.to_csv(out_file)
    
    # clean up after yourself
    shutil.rmtree(tmp_dir)

def run_classifier(Xtr, ytr, Xte, yte, scores, classes, eval_dir, eval_N,
        eval_type, n_train, n_features, i_R, i_T, i_C):

    out_dir = os.path.join(eval_dir, str(i_C), str(i_T), str(i_R))
    data_file = mkdir(os.path.join(out_dir, 'data.mat'))
    if not os.path.exists(data_file):
        t_C = classes[i_C]

        # create random split    
        y = np.ravel(ytr[:, t_C])
        split = little_cv(y, n_train[i_T])

        # balance positive and negative examples
        choices, w_tr, _ = balance_pos_neg_examples(y[split], eval_N)
        split_choice = np.flatnonzero(split)
        split_choice = split_choice[choices]
        yTr = y[split_choice]
        del y
        yTe = np.ravel(yte[:,t_C])

        # log-transform and standardize data
        log_transform = skp.FunctionTransformer(np.log1p)
        tmpXtr = log_transform.transform(Xtr[split_choice,:])
        tmpXte = log_transform.transform(Xte)
        scale = skp.StandardScaler().fit(tmpXtr)
        XTr = scale.transform(tmpXtr)
        XTe = scale.transform(tmpXte)
        del tmpXtr, tmpXte
        gc.collect()

        # choose features
        chosenFeatures = chooseFeatures(yTr,scores[split_choice,:],n_features,eval_type)
        XTr = XTr[:,chosenFeatures]
        XTe = XTe[:,chosenFeatures]

        # save the data
        nTraining = sum(yTr)
        scipy.io.savemat(data_file,{'class' : t_C,
                                    'iClass' : i_C,
                                    'nTraining' : nTraining,
                                    'iTrain' : i_T,
                                    'iRun' : i_R,
                                    'chosen_features' : chosenFeatures,
                                    'split_choice' : split_choice,
                                    'choices' : choices,
                                    'w_tr' : w_tr,
                                    'out_dir' : out_dir,
                                    'N' : eval_N,
                                    'type': eval_type})
    else:
        mat = scipy.io.loadmat(data_file)
        t_C = int(mat['class'])
        i_C = int(mat['iClass'])
        i_R = int(mat['iRun'])
        i_T = int(mat['iTrain'])
        eval_N = int(mat['N'])
        eval_type = float(mat['type'])
        split_choice = np.ravel(mat['split_choice'])
        chosenFeatures = np.ravel(mat['chosen_features'])

        log_transform = skp.FunctionTransformer(np.log1p)
        tmpXtr = log_transform.transform(Xtr[split_choice,:])
        tmpXte = log_transform.transform(Xte)
        scale = skp.StandardScaler().fit(tmpXtr)
        XTr = scale.transform(tmpXtr)[:,chosenFeatures]
        XTe = scale.transform(tmpXte)[:,chosenFeatures]
        yTr = np.ravel(ytr[split_choice, t_C])
        w_tr = np.ravel(mat['w_tr'])
        yTe = np.ravel(yte[:, t_C])
        out_dir = str(mat['out_dir'][0])
        
    results = blr.binary_log_regression(XTr, yTr, w_tr, XTe, yTe, out_dir)
    print '{:d} {:d} {:d}: {:.3f} {:.3f}'.format(i_C, i_T, i_R, results[4], results[5])
    return (t_C, i_C, sum(ytr), i_T, i_R, out_dir, eval_N, eval_type) + results


def mkdir(the_path):
     the_dir = os.path.dirname(the_path)
     if not os.path.exists(the_dir):
         os.makedirs(the_dir)
     return the_path


def chooseFeatures(y,scores,k,threshold):
    if k <= 0:
        k = scores.shape[1]
    mean_scores = np.ravel(np.mean(scores[np.nonzero(y),:], axis=0))
    z_scores = (mean_scores-np.mean(mean_scores)) / np.std(mean_scores)
    indices = np.flatnonzero(z_scores > float(threshold))

    if 1 > k > 0 and indices != []:
        features = indices[range(int(np.floor(k*scores.shape[1])))]
    elif k >= 1 and indices != []:
        features = indices[range(min(len(indices),k,scores.shape[1]))]

    return features


def balance_pos_neg_examples(y, N):
    classes = np.unique(y)
    all_choices = np.array([],  dtype=int)
    choices = np.array([], dtype=int)
    w = np.array([], dtype=int)
    for iClass in xrange(len(classes)):
        # get the examples that belong to the class
        class_members = np.flatnonzero(y == classes[iClass])
        # sample n of them
        chosen = np.random.choice(class_members, size=N, replace=True)
        uniques = np.unique(chosen)
        # record your choices
        all_choices = np.hstack([all_choices, chosen])
        choices = np.hstack([choices, uniques])
        counts = np.array([list(chosen).count(u) for u in uniques])
        w = np.hstack([w, counts])
    return np.ravel(choices), np.ravel(w), np.ravel(all_choices)


def little_cv(y, nPos):
    split = np.ones(y.size, dtype=bool)
    pos = np.flatnonzero(y)
    ignored = pos[np.random.choice(pos.size, size=pos.size-nPos, replace=False)]
    split[ignored] = False
    return split
    

def make_mmap(tmp_dir, tmp_file, data_file, var_name):
    filename = os.path.join(tmp_dir, tmp_file)
    if not os.path.exists(filename):
        h5pyfile = h5py.File(data_file,'r')
        variable = np.transpose(np.array(h5pyfile[var_name]))
        print 'putting', var_name, variable.shape, 'into', filename
        storeds = joblib.dump(variable, filename)
        del variable
        _ = gc.collect()
    return joblib.load(filename, mmap_mode='r')


def load_var(tmp_dir, tmp_file, data_file, var_name):
    h5pyfile = h5py.File(data_file,'r')
    variable = np.transpose(np.array(h5pyfile[var_name]))
    print 'loaded', var_name, variable.shape
    return variable


if __name__ == '__main__':
    if len(sys.argv) < 4:
        tmp_dir = None
    else:
        tmp_dir = sys.argv[3]
    evaluatePerformance(sys.argv[1], sys.argv[2], tmp_dir)
