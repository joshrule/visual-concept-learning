import binarylr as blr
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
import utils


def convert(obj):
    """a helper function for converting numpy objects during pickling"""
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    raise TypeError


class CategoricityEvaluator(object):
    def __init__(self, data_file, out_file):
        # Process the arguments to __init__.
        self.data_file = data_file
        self.out_file = out_file
        tmp_home = os.path.expanduser('~/.sim_tmp')
        if not os.path.exists(tmp_home):
            os.mkdir(tmp_home)
        self.tmp_dir = tempfile.mkdtemp(dir=tmp_home)

        # Load and process the parameter on-disk.
        mat = scipy.io.loadmat(data_file)

        self.tr_file = str(mat['tr_file'][0])
        self.te_file = str(mat['te_file'][0])
        self.n_train = int(np.ravel(mat['nTrain']))
        self.classes = np.ravel(mat['classes'])
        self.eval_dir = str(mat['eval_dir'][0])

        del mat, data_file, out_file

        # Load the data matrices, the actual feature values and labels.
        self.X_te = utils.make_mmap(self.tmp_dir, 'X_te.mmap', self.te_file, 'x')
        gc.collect()
        self.y_te = utils.make_mmap(self.tmp_dir, 'y_te.mmap', self.te_file, 'y')
        gc.collect()
        self.X_tr = utils.make_mmap(self.tmp_dir, 'X_tr.mmap', self.tr_file, 'x')
        gc.collect()
        self.y_tr = utils.make_mmap(self.tmp_dir, 'y_tr.mmap', self.tr_file, 'y')
        gc.collect()

        # Parallelize over up to 32 CPUs.
        self.nCPUs = min(32, mp.cpu_count())

        # Create queues to hold intermediate outputs.
        self.q0 = mp.Queue()
        self.q1 = mp.Queue()
        self.q2 = mp.Queue()

        # Create a lock to prevent races/deadlocks when printing to screen.
        self.print_lock = mp.RLock()

        gc.collect()

        # create a sub-process for each part of the evaluation.
        self.ps = [mp.Process(target=self.add_specs, args=())] + \
                  [mp.Process(target=self.specs_to_inputs, args=()) for _ in range(self.nCPUs)] + \
                  [mp.Process(target=self.run_wrapper, args=()) for _ in range(self.nCPUs)] + \
                  [mp.Process(target=self.save_data, args=())]

        # tell the log that you've finished
        print('initialized at', time.time())

    def evaluate(self):
        # reserve CPUs
        os.system('taskset -cp 0-%d %s' % (32, os.getpid()))
        os.system('taskset -p %s' %os.getpid())

        # start all the sub-processes
        for p in self.ps:
            p.start()

        # wait for them to finish
        for p in self.ps:
            p.join()

        # clean up after yourself
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def add_specs(self):
        # queue up each sub-evaluation that must be run with a proto-parameterization
        ds = ({'i_feature': i_feature, 'i_class': i_class}
              for i_class in range(self.classes.size)
              for i_feature in range(self.X_tr.shape[1]))
        for d in ds:
            self.q0.put(d)

        # queue up an end-of-iterator signal for each CPU
        for _ in range(self.nCPUs):
            self.q0.put("STOP")

        del ds
        gc.collect()

        # tell the log that you've finished
        self.print_lock.acquire()
        print('finished adding specs at', time.time())
        self.print_lock.release()

    def specs_to_inputs(self):
        # until reaching a stop signal, pop a proto-parameterization, fully parameterize it, and push it to the next queue.
        for t in iter(self.q0.get, "STOP"):
            self.q1.put(self.make_input(**t))
            gc.collect()

        # Tell the next queue and the log that you've finished.
        self.q1.put("STOP")
        self.print_lock.acquire()
        print('finished making inputs at', time.time())
        self.print_lock.release()

    def make_input(self, i_feature=None, i_class=None):
        # Determine what class to evaluate.
        target_class = int(np.ravel(self.classes[i_class]))

        # Create a random training split (i.e. choose training examples).
        y = np.ravel(self.y_tr[:, target_class])
        split_all = utils.little_cv(y, self.n_train)
        choices, w_tr, _ = utils.balance_pos_neg_examples(y[split_all], self.n_train)
        split = np.ravel(np.flatnonzero(split_all)[choices])

        # Save the parameterization.
        return {'class' : target_class,
                'i_class' : i_class,
                'n_train' : self.n_train,
                'i_feature' : i_feature,
                'split' : split,
                'choices' : choices,
                'w_tr' : w_tr}

    def run_wrapper(self):
        # until reaching a stop signal, pop a parameterization, evaluate it, and push it to the next queue.
        for d in iter(self.q1.get, "STOP"):
            # create per-parameterization data matrices
            tmpXtr = self.X_tr[np.ix_(d['split'],np.ravel(d['i_feature']))].reshape(-1,1)
            scale = skp.StandardScaler().fit(tmpXtr)
            XTr = scale.transform(tmpXtr)

            del tmpXtr
            gc.collect()

            XTe = scale.transform(self.X_te[:,d['i_feature']].reshape(-1,1))
            yTr = np.ravel(self.y_tr[np.ix_(d['split'],np.ravel(d['class']))])
            yTe = np.ravel(self.y_te[:, d['class']])

            # run the evaluation
            results = blr.binary_log_regression(XTr, yTr, d['w_tr'], XTe, yTe)
            # add key parameters to the result
            d.update(results)
            # push the result on the next queue
            self.q2.put(d)

            del scale, XTr, XTe, yTr, yTe
            gc.collect()

        # tell the next queue and the log that you've finished
        self.q2.put("STOP")
        self.print_lock.acquire()
        print('finished running classifiers at', time.time())
        self.print_lock.release()

    def save_data(self):
        # collect a record for each sub-evaluation.
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

        # tell the log that you've finished.
        self.print_lock.acquire()
        print('finished saving the data at', time.time())
        self.print_lock.release()


# the *main* event ;-)
if __name__ == '__main__':
    # tell the log that you've started
    print('started at', time.time())

    # run the evaluation
    x = CategoricityEvaluator(sys.argv[1], sys.argv[2]).evaluate()

    # tell the log that you've finished
    print('finished at', time.time())
