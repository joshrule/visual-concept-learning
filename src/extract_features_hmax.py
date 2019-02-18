# a script for extracting features from trained Caffe neural networks

import numpy as np
import sys
import caffe
import os
import csv
from os.path import basename,splitext
from math import ceil
from scipy.io import savemat
import re

# make an important path available
sim_root = '/data1/josh/ruleRiesenhuber2013/'

# add caffe to our path
caffe_root = '/home/josh/caffe/'
sys.path.insert(0, caffe_root + 'python')

def check_and_cache(model_def_file,img_list_file):
    '''cache activations unless the flag says we've already done it'''
    (img_list,unk) = splitext(basename(row[0]))
    flag = sim_root + 'evaluation/v0_2/' + img_list + '_hmax.flag'
    if not os.path.exists(flag):
        cache_activations(model_def_file,img_list_file)
        with open(flag, 'a'):
            os.utime(flag, None)
        print 'cached the images in %s' % img_list
    else:
        print 'found the flag (already cached!) for %s' % img_list

def cache_activations(model_def_file,img_list_file):
    # ensure that the necessary files exist
    model_dir = caffe_root + 'models/hmax_softmax/'
    model_def = model_dir + model_def_file
    model_weights = model_dir + 'hmax_softmax_iter_10000000.caffemodel'

    if os.path.isfile(model_weights) and os.path.isfile(model_def):
        print 'Model Found.'
    else:
        print 'Model Missing!'
        exit()

    # load the csv with our files in it
    csvfile = open(sim_root + img_list_file,'r')
    img_csv = csv.reader(csvfile, delimiter=' ')

    # initialize the gpu
    caffe.set_device(0)
    caffe.set_mode_gpu()

    # load the model in TEST mode
    net = caffe.Net(model_def, model_weights, caffe.TEST)

    count = 0
    GO = True
    batch_size = 25
    while GO:
        # run the net
        net.forward()

        # get the features
        cat_features = net.blobs['prob'].data

        for img in xrange(batch_size):
            count += 1
            
            try : # to get the filenames
                row = img_csv.next()
            except : # when there are none left
                GO = False
                break

            (img_file,_) = splitext(basename(row[0]))

            # progress update
            if count % 1000 == 0 :
                print count,' ',img_file

            # write out the features to files
            cat_file = cat_dir + img_file + '.hmax_cat_mat'
            savemat(cat_file,{'c2' : cat_features[img].reshape(2000,1)},oned_as='column')

check_and_cache('collect_features_val1.prototxt'  , 'caffe/feature_validation_images.txt')
check_and_cache('collect_features_train.prototxt' , 'caffe/evaluation_training_images.txt')
check_and_cache('collect_features_val2.prototxt'  , 'caffe/evaluation_validation_images.txt')
exit()
