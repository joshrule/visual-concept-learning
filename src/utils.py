import h5py
import os
import numpy as np
import scipy.stats

def dprime_score(y_hats,ys):
    '''Compute the d' score for a set of predictions.'''

    # compute the necessary summary stats
    truepositives = sum(h == 1 and y == 1 for h, y in zip(y_hats, ys))
    falsepositives = sum(h == 1 and y == 0 for h, y in zip(y_hats, ys))
    truenegatives = sum(h == 0 and y == 0 for h, y in zip(y_hats, ys))
    falsenegatives = sum(h == 0 and y == 1 for h, y in zip(y_hats, ys))

    numpositive = truepositives + falsenegatives
    numnegative = falsepositives + truenegatives

    tpr = float(truepositives)/float(numpositive)
    fpr = float(falsepositives)/float(numnegative)
    
    # Apply conventional corrections to avoid infinite D'.
    if (tpr == 1.0):
        tpr = float(numpositive + numnegative - 1) / float(numpositive + numnegative)
    elif (tpr == 0.0):
        tpr = 1.0 / float(numpositive + numnegative)
    if (fpr == 1.0):
        fpr = float(numpositive + numnegative - 1) / float(numpositive + numnegative)
    elif (fpr == 0.0):
        fpr = 1.0 / float(numpositive + numnegative)

    # Compute and return d'
    return scipy.stats.norm.ppf(tpr) - scipy.stats.norm.ppf(fpr)


def mkdir(the_path):
    '''Make a directory only if it doesn't already exist.'''
    the_dir = os.path.dirname(the_path)
    if not os.path.exists(the_dir):
        os.makedirs(the_dir)
    return the_path


def chooseFeatures(y,scores,k,threshold):
    '''Select a subset of the features in `scores` based on their activation strength.'''

    # Ensure we choose a valid number of features.
    if k <= 0:
        k = scores.shape[1]

    # Order the indices by z score.
    mean_scores = np.ravel(np.mean(scores[np.nonzero(y),:], axis=1))
    z_scores = (mean_scores-np.mean(mean_scores)) / np.std(mean_scores)
    z_scores_idxs = np.argsort(z_scores)

    # Compute those indices that meet our condition.
    valid_indices = np.flatnonzero(z_scores > float(threshold))

    # Remove invalid indices.
    indices = np.array([x for x in z_scores_idxs if x in valid_indices])

    # Select the features.
    if 1 > k > 0:
        return indices[range(int(np.floor(k*scores.shape[1])))]
    if k >= 1:
        return indices[range(min(len(indices),int(k),scores.shape[1]))]
    return []


def balance_pos_neg_examples(y, N):
    '''Create a weighted random split.

    For each class represented in `y`, choose `N` examples with replacement.
    '''

    # Initialize the key vectors. 
    classes = np.unique(y)
    all_choices = np.array([],  dtype=int)
    choices = np.array([], dtype=int)
    w = np.array([], dtype=int)

    # For each class...
    for iClass in range(len(classes)):
        # Get the examples that belong to the class.
        class_members = np.flatnonzero(y == classes[iClass])
        # Sample `N` of them.
        chosen = np.random.choice(class_members, size=N, replace=False)
        uniques = np.unique(chosen)
        # Record your choices.
        all_choices = np.hstack([all_choices, chosen])
        choices = np.hstack([choices, uniques])
        counts = np.array([list(chosen).count(u) for u in uniques])
        w = np.hstack([w, counts])

    # Return the set of examples, their weights, and each individual choice.
    return np.ravel(choices), np.ravel(w), np.ravel(all_choices)


def little_cv(y, nPos):
    '''Give a subset of `y` with `nPos` elements >= 0 & *all* zero elements.'''

    # Select all elements
    split = np.ones(y.size, dtype=bool)

    # Deselect all but nPos nonzero elements.
    pos = np.flatnonzero(y)
    ignored = pos[np.random.choice(pos.size, size=pos.size-nPos, replace=False)]
    split[ignored] = False

    # Return the split.
    return split
    

def make_mmap(tmp_dir, tmp_file, data_file, var_name):
    '''Create a memory-mapped file for a variable in some `data_file`.'''

    # Open the data file.
    h5pyfile = h5py.File(data_file,'r')

    # Get the shape of the variable.
    orig_shape = h5pyfile[var_name].shape
    shape = (orig_shape[1], orig_shape[0])

    # Create the memory-map file.
    filename = os.path.join(tmp_dir, tmp_file)
    mmap = np.memmap(filename, dtype=np.float32, mode='w+', shape=shape)

    # Copy the data
    print('putting', var_name, orig_shape, 'into', filename, shape)
    for idx in range(0, shape[0], 5000):
        start = idx
        stop = min(shape[0], start+5000)
        tmp_array = np.array(h5pyfile[var_name][:,start:stop]).T.astype(np.float32)
        mmap[start:stop,:] = tmp_array

    # Return the memmap object.
    return mmap
