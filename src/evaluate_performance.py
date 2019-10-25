import binarylr as blr
import gc
import gzip
import multiprocessing
import numpy as np
import os
import pandas
import pickle
import scipy.io
import shutil
import sklearn.preprocessing as skp
import sys
import tempfile
import time
import utils


class PerformanceEvaluator(object):
    def __init__(self, data_file, out_file, tmp_dir=None):
        # These are the arguments to __init__.
        self.data_file = data_file
        self.out_file = out_file
        if tmp_dir is None: # The extra logic lets you create the tmp data in advance.
            tmp_home = os.path.expanduser('~/.sim_tmp')
            if not os.path.exists(tmp_home):
                os.mkdir(tmp_home)
            self.tmp_dir = tempfile.mkdtemp(dir=tmp_home)
        else:
            self.tmp_dir = tmp_dir

        # Load and process the parameter on-disk.
        mat = scipy.io.loadmat(data_file)

        self.tr_file = str(mat['tr_file'][0])
        self.te_file = str(mat['te_file'][0])
        self.score_file = str(mat['score_file'][0])

        self.eval_dir = str(mat['eval_dir'][0])
        self.eval_type = float(mat['thresh'].item())

        self.n_train = np.ravel(mat['nTrain'])
        self.n_runs = int(mat['nRuns'])
        self.n_features = float(mat['nFeatures'].item())
        self.classes = np.ravel(mat['classes'])
        self.small = bool(np.ravel(mat['small']))

        del mat, data_file, out_file, tmp_dir

        # Load the data matrices, the actual feature values and labels.
        self.X_te = utils.make_mmap(self.tmp_dir, 'X_te.mmap', self.te_file, 'x')
        gc.collect()
        self.y_te = utils.make_mmap(self.tmp_dir, 'y_te.mmap', self.te_file, 'y')
        gc.collect()
        self.X_tr = utils.make_mmap(self.tmp_dir, 'X_tr.mmap', self.tr_file, 'x')
        gc.collect()
        self.y_tr = utils.make_mmap(self.tmp_dir, 'y_tr.mmap', self.tr_file, 'y')
        gc.collect()
        if self.score_file == self.tr_file:
            self.scores = None
        else:
            self.scores = make_mmap(self.tmp_dir, 'scores.mmap', self.score_file, 'x')
            gc.collect()

        # Parallelize over up to 32 CPUs.
        self.nCPUs = min(32, multiprocessing.cpu_count())

        # These queues are used to hold intermediate outputs.
        self.q0 = multiprocessing.Queue()
        self.q1 = multiprocessing.Queue()
        self.q2 = multiprocessing.Queue()

        # This Lock is used to prevent races/deadlocks when printing to screen.
        self.print_lock = multiprocessing.RLock()

        gc.collect()

        # dump each problem spec onto a queue
        self.p0 = multiprocessing.Process(target=self.add_specs, args=())
        
        # pull specs from queue, transform into inputs, and dump onto another queue
        self.p1s = [multiprocessing.Process(target=self.specs_to_inputs, args=())
                    for _ in range(self.nCPUs)]

        # pull inputs from queue, transform into outputs, and dump onto another queue
        self.p2s = [multiprocessing.Process(target=self.run_wrapper, args=())
               for _ in range(self.nCPUs)]

        # pull outputs from queue and transform into final table
        self.p3 = multiprocessing.Process(target=self.make_table, args=())
    
        # tell the log that you've finished
        print('initialized at', time.time())

    def evaluate(self):
        # reserve CPUs
        os.system('taskset -cp 0-%d %s' % (32, os.getpid()))
        os.system('taskset -p %s' %os.getpid())

        # start all the sub-processes
        self.p0.start()
        for p in self.p1s:
            p.start()
        for p in self.p2s:
            p.start()
        self.p3.start()

        # wait for them to finish
        self.p0.join()
        for p in self.p1s:
            p.join()
        for p in self.p2s:
            p.join()
        self.p3.join()

        # clean up after yourself
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def add_specs(self):
        # queue up each sub-evaluation that must be run with a proto-parameterization
        ts = ((i_run, i_train, i_class)
              for i_run in range(self.n_runs) 
              for i_train in range(self.n_train.size)
              for i_class in range(self.classes.size))
        for t in ts:
            self.q0.put(t)

        # queue up an end-of-iterator signal for each CPU
        for _ in range(self.nCPUs):
            self.q0.put("STOP")

        del ts
        self.print_lock.acquire()

        # tell the log that you've finished
        print('finished adding specs at', time.time())
        self.print_lock.release()

    def specs_to_inputs(self):
        # until reaching a stop signal, pop a proto-parameterization, fully parameterize it, and push it to the next queue.
        for t in iter(self.q0.get, "STOP"):
            self.q1.put(self.make_input(*t))
            gc.collect()

        # tell the next queue and the log that you've finished
        self.q1.put("STOP")
        self.print_lock.acquire()
        print('finished making inputs at', time.time())
        self.print_lock.release()

    def run_wrapper(self):
        # until reaching a stop signal, pop a parameterization, evaluate it, and push it to the next queue.
        for data_file in iter(self.q1.get, "STOP"):
            # read the parameterization
            with gzip.open(data_file, 'rb') as fd:
                input_dict = pickle.load(fd)
            t_C = int(input_dict['class'])
            i_C = int(input_dict['iClass'])
            i_R = int(input_dict['iRun'])
            i_T = int(input_dict['iTrain'])
            eval_type = float(input_dict['type'])
            split_choice = np.ravel(input_dict['split_choice'])
            out_dir = str(input_dict['out_dir'])
            w_tr = np.ravel(input_dict['w_tr'])
            small = bool(np.ravel(input_dict['small']))
            del input_dict

            # create per-parameterization data matrices
            tmpXtr = self.X_tr[split_choice,:]
            tmpXte = self.X_te
            scale = skp.StandardScaler().fit(tmpXtr)
            XTr = scale.transform(tmpXtr)
            del tmpXtr
            gc.collect()
            XTe = scale.transform(tmpXte)
            yTr = np.ravel(self.y_tr[np.ix_(split_choice,np.ravel(t_C))])
            yTe = np.ravel(self.y_te[:, t_C])
            del scale, tmpXte

            # run the evaluation
            results = blr.binary_log_regression(XTr, yTr, w_tr, XTe, yTe, out_dir, small)
            # add key parameters to the result
            results.update({"class": t_C, "iClass": i_C, "iTrain": i_T, "iRun": i_R,
                "nTraining": sum(yTr), "out_dir": out_dir, "eval_type": eval_type})
            # push the result on the next queue
            self.q2.put(results)

            del t_C, i_C, i_R, i_T, eval_type, split_choice
            del out_dir, w_tr, XTr, XTe, yTr, yTe
            gc.collect()

        # tell the next queue and the log that you've finished
        self.q2.put("STOP")
        self.print_lock.acquire()
        print('finished running classifiers at', time.time())
        self.print_lock.release()

    def make_table(self):
        # collect a record for each sub-evaluation.
        records = []
        stops = 0
        while stops < self.nCPUs:
            record = self.q2.get()
            if record == 'STOP':
                stops += 1
            else:
                records.append(record)

        # perform a bit of post-processing and write out the results two different ways
        if not self.small:
            for record in records:
                del record['features']
            with gzip.open(os.path.join(self.eval_dir, "evaluation.pkl"), 'wb') as fd:
                pickle.dump(records, fd)
            for record in records:
                del record['model']
        df = pandas.DataFrame.from_records(records)
        df.to_csv(self.out_file)
        del df

        # tell the log that you've finished
        self.print_lock.acquire()
        print('finished making the table at', time.time())
        self.print_lock.release()

    def make_input(self, i_R, i_T, i_C):
        out_dir = os.path.join(self.eval_dir, str(i_C), str(i_T), str(i_R))
        data_file = utils.mkdir(os.path.join(out_dir, 'data.pkl'))
        if not os.path.exists(data_file):
            t_C = self.classes[i_C]

            # create random split    
            y = np.ravel(self.y_tr[:, t_C])
            split = utils.little_cv(y, self.n_train[i_T])
            choices, w_tr, _ = utils.balance_pos_neg_examples(y[split], self.n_train[i_T])
            split_choice = np.ravel(np.flatnonzero(split)[choices])
            yTr = y[split_choice]
            del y, split
            gc.collect()

            # save data
            input_dict =  {
                    'class' : t_C,
                    'iClass' : i_C,
                    'nTraining' : sum(yTr),
                    'iTrain' : i_T,
                    'iRun' : i_R,
                    'split_choice' : split_choice,
                    'choices' : choices,
                    'w_tr' : w_tr,
                    'out_dir' : out_dir,
                    'small' : self.small,
                    'type': self.eval_type
                    }
            with gzip.open(data_file, 'wb') as fd:
                pickle.dump(input_dict, fd)
        return data_file


# the *main* event ;-)
if __name__ == '__main__':
    # tell the log that you've started
    print('started at', time.time())

    # determine whether you already have a tmp_dir
    if len(sys.argv) < 4:
        tmp_dir = None
    else:
        tmp_dir = sys.argv[3]

    # run the evaluation
    PerformanceEvaluator(sys.argv[1], sys.argv[2], tmp_dir).evaluate()

    # tell the log that you've finished
    print('finished at', time.time())
