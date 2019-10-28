# a script for extracting features from a trained Caffe neural network
# much stealing from:
#   http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb

import caffe
from math import ceil
import numpy as np
import os
from os.path import basename,splitext
import pandas as pd
import re
from scipy.io import savemat
import sys
import time

# make an important path available
sim_root = '/data1/josh/concept_learning/'

# add caffe to our path
caffe_root = '/home/josh/caffe/'
sys.path.insert(0, caffe_root + 'python')

def check_and_cache(model_def_file,img_list_file,modelname,model_dir):
    '''cache activations unless the flag says we've already done it'''
    (img_list,unk) = splitext(basename(img_list_file))
    flag = sim_root + 'evaluation/v0_2/' + img_list + '_' + modelname + '.flag'
    if not os.path.exists(flag):
        cache_activations(model_def_file,img_list_file,model_dir)
        with open(flag, 'a'):
            os.utime(flag, None)
        print 'cached the images in %s' % img_list
    else:
        print 'found the flag (already cached!) for %s' % img_list


def save_activation(filename, activation):
    if not os.path.exists(filename):
        savemat(filename,
                {'c2' : np.ravel(activation)},
                oned_as='column',
                appendmat=False)


def process_hmax_net(net, csvfiles):
    batch_size = 25
    img_num = 0
    while img_num < len(csvfiles):
        # run the net
        net.forward()

        # get the features -- for HMAX, we're only interested in one layer.
        cat_features = net.blobs['prob'].data

        for img in xrange(batch_size):
            
            # progress update
            if img_num % 1000 == 0:
                print '{:d}/{:d}: {} - {}'.format(img_num,len(csvfiles),str(time.time()),csvfiles[img_num])

            (img_name,_) = splitext(csvfiles[img_num])

            # write out the features to files
            save_activation(img_name + '.hmax_cat_mat', cat_features[img])

            img_num += 1


def process_googlenet_net(net, csvfiles):
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
            image = image[int(crop[0]):int(crop[1]),int(crop[2]):int(crop[3]),:]
            transformed_image = transformer.preprocess('data', image)
            new_blob[i,:,:,:] = transformed_image

        # add data to network, classify, and pull the features
        net.blobs['data'].data[...] = new_blob
        net.forward()
        cat_features = net.blobs['prob'].data
        cat2_features = net.blobs['loss3/classifier'].data
        gen_features = net.blobs['pool5/7x7_s1'].data
        gen2_features = net.blobs['loss2/ave_pool'].data
        gen3_features = net.blobs['loss1/ave_pool'].data

        # write out the features to files
        for i,imgfile in enumerate(imgfiles):

            (img_name,_) = splitext(imgfile)

            save_activation(img_name + '.googlenet_cat_mat', cat_features[i])
            save_activation(img_name + '.googlenet_cat2_mat', cat2_features[i])
            save_activation(img_name + '.googlenet_gen_mat', gen_features[i])
            save_activation(img_name + '.googlenet_gen2_mat', gen2_features[i])
            save_activation(img_name + '.googlenet_gen3_mat', gen3_features[i])

        img_num += cur_batch_size

def cache_activations(model_def_file,img_list_file,model_dir):
    '''cache activations:

    Write to disk the activations of several layers in a given model for a
    given list of images.
    '''
    # ensure that the necessary files exist
    model_dir = caffe_root + 'models/' + model_dir + '/'
    model_def = model_dir + model_def_file
    model_weights = model_dir + model_dir + '_iter_10000000.caffemodel'

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

    if model_dir == 'hmax_softmax':
       process_hmax_net(net, csvfiles) 
    else:
        process_googlenet_net(net, csvfiles)

check_and_cache('deploy.prototxt', 'caffe/evaluation_validation_images.txt', 'googlenet', 'maxlab_googlenet')
check_and_cache('deploy.prototxt', 'caffe/evaluation_training_images.txt', 'googlenet', 'maxlab_googlenet')
check_and_cache('deploy.prototxt', 'caffe/feature_training_images.txt', 'googlenet', 'maxlab_googlenet')
if false:
    check_and_cache('collect_features_val1.prototxt'  , 'caffe/feature_validation_images.txt', 'hmax', 'hmax_softmax')
    check_and_cache('collect_features_train.prototxt' , 'caffe/evaluation_training_images.txt', 'hmax', 'hmax_softmax')
    check_and_cache('collect_features_val2.prototxt'  , 'caffe/evaluation_validation_images.txt' 'hmax', 'hmax_softmax')
exit()
