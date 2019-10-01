import binarylr as blr
import utils
import gc
import gzip
import json
import multiprocessing as mp
import numpy as np
import os
import scipy.io
import shutil
import sklearn.preprocessing as skp
import sys
import tempfile
import time

def convert(obj):
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    raise TypeError

class CategoricityEvaluator(object):
    def __init__(self, data_file, out_file):
        # Assume you've been given data file containing
        # - x,y for training data
        # - x,y for testing data
        # - the list of classes to test
        # - the number of training examples
        self.data_file = data_file
        self.out_file = out_file

        tmp_home = os.path.expanduser('~/.sim_tmp')
        if not os.path.exists(tmp_home):
            os.mkdir(tmp_home)
        self.tmp_dir = tempfile.mkdtemp(dir=tmp_home)

        mat = scipy.io.loadmat(data_file)

        self.tr_file = str(mat['tr_file'][0])
        self.te_file = str(mat['te_file'][0])
        self.n_train = int(np.ravel(mat['nTrain']))
        self.classes = np.ravel(mat['classes'])

        self.eval_dir = str(mat['eval_dir'][0])

        del mat, data_file, out_file

        self.X_te = utils.make_mmap(self.tmp_dir, 'X_te.mmap', self.te_file, 'x')
        gc.collect()
        self.y_te = utils.make_mmap(self.tmp_dir, 'y_te.mmap', self.te_file, 'y')
        gc.collect()
        self.X_tr = utils.make_mmap(self.tmp_dir, 'X_tr.mmap', self.tr_file, 'x')
        gc.collect()
        self.y_tr = utils.make_mmap(self.tmp_dir, 'y_tr.mmap', self.tr_file, 'y')
        gc.collect()

        self.nCPUs = min(32, mp.cpu_count())

        self.q0 = mp.Queue()
        self.q1 = mp.Queue()
        self.q2 = mp.Queue()
        self.print_lock = mp.RLock()

        gc.collect()

        self.ps = [mp.Process(target=self.add_specs, args=())] + \
                  [mp.Process(target=self.specs_to_inputs, args=()) for _ in range(self.nCPUs)] + \
                  [mp.Process(target=self.run_wrapper, args=()) for _ in range(self.nCPUs)] + \
                  [mp.Process(target=self.save_data, args=())]

    def evaluate(self):
        # reserve CPUs
        os.system('taskset -cp 0-%d %s' % (32, os.getpid()))
        os.system('taskset -p %s' %os.getpid())

        # start all the processes
        for p in self.ps:
            p.start()

        # wait for them to finish
        for p in self.ps:
            p.join()

        # clean up after yourself
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def add_specs(self):
        ds = ({'i_feature': i_feature, 'i_class': i_class}
              for i_class in range(self.classes.size)
              for i_feature in range(self.X_tr.shape[1]))

        for d in ds:
            self.q0.put(d)

        for _ in range(self.nCPUs):
            self.q0.put("STOP")

        del ds
        gc.collect()

        self.print_lock.acquire()
        print('finished adding specs at', time.time())
        self.print_lock.release()

    def specs_to_inputs(self):
        for t in iter(self.q0.get, "STOP"):
            self.q1.put(self.make_input(**t))
            gc.collect()

        self.q1.put("STOP")

        self.print_lock.acquire()
        print('finished making inputs at', time.time())
        self.print_lock.release()

    def make_input(self, i_feature=None, i_class=None):
        target_class = int(np.ravel(self.classes[i_class]))

        # create random split (i.e. choose training examples)    
        y = np.ravel(self.y_tr[:, target_class])
        split_all = utils.little_cv(y, self.n_train)
        choices, w_tr, _ = utils.balance_pos_neg_examples(y[split_all], self.n_train)
        split = np.ravel(np.flatnonzero(split_all)[choices])

        # save data
        return {'class' : target_class,
                'i_class' : i_class,
                'n_train' : self.n_train,
                'i_feature' : i_feature,
                'split' : split,
                'choices' : choices,
                'w_tr' : w_tr}

    def run_wrapper(self):
        for d in iter(self.q1.get, "STOP"):
            tmpXtr = self.X_tr[np.ix_(d['split'],np.ravel(d['i_feature']))].reshape(-1,1)
            scale = skp.StandardScaler().fit(tmpXtr)
            XTr = scale.transform(tmpXtr)

            del tmpXtr
            gc.collect()

            XTe = scale.transform(self.X_te[:,d['i_feature']].reshape(-1,1))
            yTr = np.ravel(self.y_tr[np.ix_(d['split'],np.ravel(d['class']))])
            yTe = np.ravel(self.y_te[:, d['class']])

            results = blr.binary_log_regression(XTr, yTr, d['w_tr'], XTe, yTe)
            d.update(results)
            self.q2.put(d)

            del scale, XTr, XTe, yTr, yTe
            gc.collect()

        self.q2.put("STOP")

        self.print_lock.acquire()
        print('finished running classifiers at', time.time())
        self.print_lock.release()

    def save_data(self):
        records = []
        stops = 0
        while stops < self.nCPUs:
            record = self.q2.get()
            if record == 'STOP':
                stops += 1
            else:
                record['split'] = list(np.ravel(record['split']))
                record['choices'] = list(np.ravel(record['choices']))
                record['w_tr'] = list(np.ravel(record['w_tr']))
                records.append(record)

        # write out gzipped json
        with gzip.open(self.out_file, 'wt', encoding="utf-8") as f:
               json.dump(records, f, default=convert)

        self.print_lock.acquire()
        print('finished saving the data at', time.time())
        self.print_lock.release()

if __name__ == '__main__':
    print('started at', time.time())
    x = CategoricityEvaluator(sys.argv[1], sys.argv[2])
    print('initialized at', time.time())
    x.evaluate()
    print('finished at', time.time())
