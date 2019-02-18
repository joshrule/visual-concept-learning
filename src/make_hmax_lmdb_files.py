# boldly stolen from:
# http://deepdish.io/2015/04/28/creating-lmdb-in-python/
# http://research.beenfrog.com/code/2015/12/30/write-read-lmdb-example.html

import caffe
import csv
import lmdb
import numpy as np
import os.path
import re
import scipy.io
from math import ceil
from random import shuffle

def create_the_lmdb_file(csvfilename,lmdbfilename):

    if os.path.exists(lmdbfilename):
        print '%s already exists, skipping!' % lmdbfilename
    else:
        # read in the list of images
        with open(csvfilename, 'r') as csvfile:
            data = csv.reader(csvfile, delimiter=' ')
            files_and_labels = zip(*[(row[0],row[1]) for row in data])
            files = list(files_and_labels[0])
            labels = list(files_and_labels[1])
        N = len(labels)
        indices = range(N)
        shuffle(indices) # crucial for good training
        mapsize = ceil(N * 3200 * 8 * 1.1) # Makes the db 10% larger than needed.
        env = lmdb.open(lmdbfilename, map_size=mapsize)

        regex = re.compile('JPEG');
        with env.begin(write=True) as txn:
            for i,idx in enumerate(indices):
                if i%10000 == 0:
                    print i
                hmax_file = regex.sub('hmax_gen_mat',files[idx])
                mat = scipy.io.loadmat(hmax_file) 
                c2 = mat['c2'].astype(np.float64,casting='equiv')
                X = np.reshape(c2,(1,1,3200))
                y = int(labels[idx])
                datum = caffe.io.array_to_datum(X, y)
                txn.put('{:0>8d}'.format(i), datum.SerializeToString())

        print '%s created' % lmdbfilename

create_the_lmdb_file('/data1/josh/ruleRiesenhuber2013/caffe/feature_validation_images.txt','/data2/image_sets/image_net/feat_val_hmax_lmdb')
create_the_lmdb_file('/data1/josh/ruleRiesenhuber2013/caffe/feature_training_images.txt','/data2/image_sets/image_net/feat_train_hmax_lmdb')
