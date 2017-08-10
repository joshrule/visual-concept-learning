# a script for extracting features from a trained Caffe neural network
# much stealing from:
#   http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb

import caffe
import numpy as np
import os
import pandas as pd
import re
import sys
import time
from os.path import basename,splitext
from math import ceil
from scipy.io import savemat

# make an important path available
sim_root = '/data1/josh/ruleRiesenhuber2013/'

# add caffe to our path
caffe_root = '/home/josh/caffe/'
sys.path.insert(0, caffe_root + 'python')

def check_and_cache(model_def_file,img_list_file):
    '''cache activations unless the flag says we've already done it'''
    (img_list,unk) = splitext(basename(img_list_file))
    flag = sim_root + 'evaluation/v0_2/' + img_list + '_googlenet.flag'
    if not os.path.exists(flag):
        cache_activations(model_def_file,img_list_file)
        with open(flag, 'a'):
            os.utime(flag, None)
        print 'cached the images in %s' % img_list
    else:
        print 'found the flag (already cached!) for %s' % img_list

def cache_activations(model_def_file,img_list_file):
    '''actually perform the caching of features'''
    # ensure that the necessary files exist
    model_dir = caffe_root + 'models/maxlab_googlenet/'
    model_def = model_dir + model_def_file
    model_weights = model_dir + 'maxlab_googlenet_iter_10000000.caffemodel'

    if os.path.isfile(model_weights) and os.path.isfile(model_def):
        print 'Model Found.'
    else:
        print 'Model Missing!'
        exit()

    # load the csv with our files in it
    csvfile = sim_root + img_list_file
    csvdata = pd.read_csv(csvfile, sep=' ', header=None, names=['file','label'])
    csvfiles = csvdata.file.values

    # initialize the gpu
    caffe.set_device(0)
    caffe.set_mode_gpu()

    # load the model in TEST mode
    net = caffe.Net(model_def, model_weights, caffe.TEST)

    # load the mean pixel value from training
    pixelfile = sim_root + 'caffe/feature_training_images_means.csv'
    pixeldata = pd.read_csv(pixelfile)
    mu = np.array([pixeldata['Blue'].iloc[0], pixeldata['Green'].iloc[0], pixeldata['Red'].iloc[0]])

    # setup the pixel transformer
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
    transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

    batch_size = 8
    crop_size = (256,256)

    img_num = 0
    while img_num < len(csvfiles):

        if img_num % 1000 == 0:
            print '{:d}/{:d}: {} - {}'.format(img_num,len(csvfiles),str(time.time()),csvfiles[img_num])

        cur_batch_size = min(batch_size,len(csvfiles)-img_num)
        imgfiles = csvfiles[img_num:(img_num+cur_batch_size)]
        new_blob = np.zeros((batch_size,3,crop_size[0],crop_size[1]))

        # preprocess image
        for i,imgfile in enumerate(imgfiles):
            image = caffe.io.load_image(imgfile)
            img_shape = image.shape
            crop = [np.floor((img_shape[0]-crop_size[0])/2.0),
                    np.floor((img_shape[0]-crop_size[0])/2.0)+crop_size[0],
                    np.floor((img_shape[1]-crop_size[1])/2.0),
                    np.floor((img_shape[1]-crop_size[1])/2.0)+crop_size[1]]
            image = image[crop[0]:crop[1],crop[2]:crop[3],:]
            transformed_image = transformer.preprocess('data', image)
            new_blob[i,:,:,:] = transformed_image

        # add data to network, classify, and pull the features
        net.blobs['data'].data[...] = new_blob
        net.forward()
        cat_features = net.blobs['prob'].data
        gen_features = net.blobs['pool5/7x7_s1'].data

        # write out the features to files
        for i,imgfile in enumerate(imgfiles):

            (img_name,_) = splitext(imgfile)

            cat_file = img_name + '.googlenet_cat_mat'
            savemat(cat_file,{'c2' : cat_features[i].reshape(2000,1)},oned_as='column',appendmat=False)

            gen_file = img_name + '.googlenet_gen_mat'
            savemat(gen_file,{'c2' : gen_features[i].reshape(4096,1)},oned_as='column',appendmat=False)

        img_num += cur_batch_size

check_and_cache('deploy.prototxt', 'caffe/evaluation_validation_images.txt')
check_and_cache('deploy.prototxt', 'caffe/evaluation_training_images.txt')
check_and_cache('deploy.prototxt', 'caffe/feature_training_images.txt')
exit()
