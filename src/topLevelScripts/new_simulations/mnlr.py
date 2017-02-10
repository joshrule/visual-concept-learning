# stolen from http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/brewing-logreg.ipynb
import time
start_time = time.time()

import os
caffe_home = os.environ['CAFFE_HOME']

import sys
sys.path.insert(0, os.path.join(caffe_home, 'python'))

import caffe
import h5py
import numpy as np
import pickle
import scipy.io
import shutil
import sklearn
import tempfile
import fnmatch

gpu_to_use = int(sys.argv[1])
dirname = sys.argv[2]

# PART 1: define the top_k function
def top_k(features,labels,k):
    nExamples = len(labels)
    topKs = np.zeros((nExamples))
    for item in range(nExamples):
        idxs = np.argsort(features[item,:])
        topKs[item] = int(labels[item] in idxs[-k:])
    return np.mean(topKs)

# PART 2: write out the data for Caffe

## write the training data
train_f = []
with open(os.path.join(dirname, 'train.txt'), 'w') as f:
    for file in os.listdir(dirname):
            if fnmatch.fnmatch(file, '*train*.h5'):
                file_name = os.path.join(dirname,file)
                f.write(file_name + '\n')
                train_f += [h5py.File(file_name,'r')]

## write the testing data - fixed in size
test_filename = os.path.join(dirname, 'test.h5')

with open(os.path.join(dirname, 'test.txt'), 'w') as f:
    f.write(test_filename + '\n')

test_f = h5py.File(test_filename,'r')

# PART 3: Define the network
from caffe import layers as L
from caffe import params as P

def logreg(hdf5, batch_size):
    # logistic regression: data, dropout, inner product, and 1000-way softmax
    n = caffe.NetSpec()
    n.data, n.label = L.HDF5Data(batch_size=batch_size, source=hdf5, ntop=2)
    n.dropout = L.Dropout(n.data, dropout_ratio=0.5)
    n.ip1 = L.InnerProduct(n.dropout, num_output=1000, weight_filler=dict(type='xavier'))
    n.prob = L.Softmax(n.ip1)
    n.accuracy1 = L.Accuracy(n.prob, n.label)
    n.accuracy5 = L.Accuracy(n.prob, n.label, top_k = 5)
    n.loss = L.SoftmaxWithLoss(n.ip1, n.label)
    return n.to_proto()

train_net_path = os.path.join(dirname, 'logreg_auto_train.prototxt')
with open(train_net_path, 'w') as f:
    f.write(str(logreg(os.path.join(dirname, 'train.txt'), 100)))

test_net_path = os.path.join(dirname, 'logreg_auto_test.prototxt')
with open(test_net_path, 'w') as f:
    f.write(str(logreg(os.path.join(dirname, 'test.txt'), 100)))

# PART 4: Define the solver
from caffe.proto import caffe_pb2

def solver(train_net_path, test_net_path, n_examples, batch_size):
    min_max_iter = 10000
    target_epochs = 100
    # n_iters should be 10000 iters or 100 epochs, whichever comes later
    n_iters = max(target_epochs*n_examples/batch_size,min_max_iter)

    s = caffe_pb2.SolverParameter()
    s.train_net = train_net_path     # where train network is
    s.test_net.append(test_net_path) # where test network is
    s.test_interval = n_iters/25     # test after every 4 epochs
    s.test_iter.append(500)          # Test 500 batches each time we test.
    s.max_iter = n_iters             # the number of training iterations
    s.base_lr = 0.01                 # the initial learning rate for SGD.
    s.lr_policy = 'step'             # lr <- lr*gamma every stepsize iters
    s.gamma = 0.9                    #
    s.stepsize = n_iters/25          #
    s.momentum = 0.9                 # weighted avg of current and previous gradients
    s.weight_decay = 5e-3            # regularizes learning to help prevent overfitting
    s.display = min(n_iters/25,1000) # display outputs every so often
    s.snapshot = n_iters             # snapshot at the end
    s.snapshot_prefix = os.path.join(dirname, 'train')
    s.solver_mode = caffe_pb2.SolverParameter.GPU

    return s

solver_path = os.path.join(dirname, 'logreg_solver.prototxt')
with open(solver_path, 'w') as f:
    f.write(str(solver(train_net_path, test_net_path, sum([len(f['data']) for f in train_f]), 100)))

# PART 5: run the solver 
print 'GPU used: {:d}'.format(gpu_to_use)
caffe.set_mode_gpu()
caffe.set_device(gpu_to_use)
solver = caffe.get_solver(solver_path)
solver.solve()

# PART 6: test the solver
labels = test_f['label'].value
batch_size = solver.test_nets[0].blobs['data'].num
test_iters = int(len(labels) / batch_size)
features = np.zeros((len(labels),1000))
for i in range(test_iters):
    solver.test_nets[0].forward()
    start = i*batch_size
    stop = min(len(labels),start + batch_size)
    features[start:stop,:] =  solver.test_nets[0].blobs['prob'].data[0:(stop-start),:]

# PART 7: save results
os.remove(os.path.join(dirname, 'train.txt'))
os.remove(os.path.join(dirname, 'train.h5'))
os.remove(os.path.join(dirname, 'test.txt'))
os.remove(os.path.join(dirname, 'test.h5'))

result_file = os.path.join(dirname, 'results.mat')
top_1 = top_k(features,labels,1)
top_5 = top_k(features,labels,5)

scipy.io.savemat(result_file,{'features' : features,
                              'top_1' : top_1,
                              'top_5' : top_5})

# PART 8: report the results
stop_time = time.time()
print 'top-1: {:.3f}'.format(top_1)
print 'top-5: {:.3f}'.format(top_5)
print 'time to train and test classifier: {:.2f}s'.format(stop_time-start_time)
