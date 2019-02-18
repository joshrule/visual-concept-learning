import h5py
import os
import numpy as np
import scipy.stats

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


def mkdir(the_path):
     the_dir = os.path.dirname(the_path)
     if not os.path.exists(the_dir):
         os.makedirs(the_dir)
     return the_path


def chooseFeatures(y,scores,k,threshold):
    if k <= 0:
        k = scores.shape[1]
    mean_scores = np.ravel(np.mean(scores[np.nonzero(y),:], axis=1))
    z_scores = (mean_scores-np.mean(mean_scores)) / np.std(mean_scores)
    # order the indices by z score
    z_scores_idxs = np.argsort(z_scores)
    # compute those indices that meet our condition
    valid_indices = np.flatnonzero(z_scores > float(threshold))
    # remove invalid indices
    indices = np.array([x for x in z_scores_idxs if x in valid_indices])
    # select the features
    if 1 > k > 0:
        return indices[range(int(np.floor(k*scores.shape[1])))]
    if k >= 1:
        return indices[range(min(len(indices),int(k),scores.shape[1]))]
    return []


def balance_pos_neg_examples(y, N):
    classes = np.unique(y)
    all_choices = np.array([],  dtype=int)
    choices = np.array([], dtype=int)
    w = np.array([], dtype=int)
    for iClass in xrange(len(classes)):
        # get the examples that belong to the class
        class_members = np.flatnonzero(y == classes[iClass])
        # sample n of them
        chosen = np.random.choice(class_members, size=N, replace=False)
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
    h5pyfile = h5py.File(data_file,'r')

    # get shape of variable
    orig_shape = h5pyfile[var_name].shape
    shape = (orig_shape[1], orig_shape[0])

    filename = os.path.join(tmp_dir, tmp_file)
    mmap = np.memmap(filename, dtype=np.float32, mode='w+', shape=shape)
    print 'putting', var_name, orig_shape, 'into', filename, shape
    for idx in range(0, shape[0], 5000):
        start = idx
        stop = min(shape[0], start+5000)
        tmp_array = np.array(h5pyfile[var_name][:,start:stop]).T.astype(np.float32)
        mmap[start:stop,:] = tmp_array
    return mmap
