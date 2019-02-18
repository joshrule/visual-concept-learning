import binarylr as blr
import gc
import multiprocessing
import numpy as np
import os
import pandas
import scipy.io
import shutil
import sklearn.preprocessing as skp
import sys
import tempfile
import time
import utils


class PerformanceEvaluator(object):
    def __init__(self, data_file, out_file, tmp_dir=None):
        self.data_file = data_file
        self.out_file = out_file
        if tmp_dir is None:
            tmp_home = os.path.expanduser('~/.sim_tmp')
            if not os.path.exists(tmp_home):
                os.mkdir(tmp_home)
            self.tmp_dir = tempfile.mkdtemp(dir=tmp_home)
        else:
            self.tmp_dir = tmp_dir

        mat = scipy.io.loadmat(data_file)

        self.tr_file = str(mat['tr_file'][0])
        self.te_file = str(mat['te_file'][0])
        self.score_file = str(mat['score_file'][0])

        self.eval_dir = str(mat['eval_dir'][0])
        self.eval_N = int(mat['eval_N'])
        self.eval_type = float(np.asscalar(mat['thresh']))
        self.permute = bool(mat.get('permute',False))

        self.n_train = np.ravel(mat['nTrain'])
        self.n_runs = int(mat['nRuns'])
        self.n_features = float(np.asscalar(mat['nFeatures']))
        self.classes = np.ravel(mat['classes'])

        del mat, data_file, out_file, tmp_dir

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

        self.nCPUs = min(32, multiprocessing.cpu_count())

        self.q0 = multiprocessing.Queue()
        self.q1 = multiprocessing.Queue()
        self.q2 = multiprocessing.Queue()

        self.q0_stops = multiprocessing.Value('i', 0, lock=True)
        self.q1_stops = multiprocessing.Value('i', 0, lock=True)
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


    def evaluatePerformance(self):
        # os.system('taskset -cp 0-%d %s' % (multiprocessing.cpu_count(), os.getpid()))
        os.system('taskset -cp 0-%d %s' % (32, os.getpid()))
        os.system('taskset -p %s' %os.getpid())

        # start all the processes
        self.p0.start()
        for p in self.p1s:
            p.start()
        for p in self.p2s:
            p.start()
        self.p3.start()

        self.p0.join()
        for p in self.p1s:
            p.join()
        for p in self.p2s:
            p.join()
        self.p3.join()

        # clean up after yourself
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def add_specs(self):
        ts = ((i_run, i_train, i_class)
              for i_run in xrange(self.n_runs) 
              for i_train in xrange(self.n_train.size)
              for i_class in xrange(self.classes.size))
        for t in ts:
            self.q0.put(t)

        for _ in xrange(self.nCPUs):
            self.q0.put("STOP")

        del ts
        self.print_lock.acquire()
        print 'finished adding specs at', time.time()
        self.print_lock.release()

    def specs_to_inputs(self):
        for t in iter(self.q0.get, "STOP"):
            self.q1.put(self.make_input(*t))
            gc.collect()

        self.q1.put("STOP")
        self.print_lock.acquire()
        print 'finished making inputs at', time.time()
        self.print_lock.release()

    def run_wrapper(self):
        for data_file in iter(self.q1.get, "STOP"):
            mat = scipy.io.loadmat(data_file)
            t_C = int(mat['class'])
            i_C = int(mat['iClass'])
            i_R = int(mat['iRun'])
            i_T = int(mat['iTrain'])
            eval_N = int(mat['N'])
            eval_type = float(mat['type'])
            split_choice = np.ravel(mat['split_choice'])
            chosenFeatures =  np.ravel(mat['chosen_features'])
            out_dir = str(mat['out_dir'][0])
            w_tr = np.ravel(mat['w_tr'])
            del mat

            # TODO: reintroduce log transform 
            # log_transform = skp.FunctionTransformer(np.log1p)
            # tmpXtr = log_transform.transform(self.X_tr[np.ix_(split_choice,chosenFeatures)])
            # tmpXte = log_transform.transform(self.X_te[:,chosenFeatures])
            tmpXtr = self.X_tr[np.ix_(split_choice,chosenFeatures)]
            tmpXte = self.X_te[:,chosenFeatures]
            scale = skp.StandardScaler().fit(tmpXtr)
            XTr = scale.transform(tmpXtr)

            # del log_transform, tmpXtr
            del tmpXtr
            gc.collect()

            XTe = scale.transform(tmpXte)
            yTr = np.ravel(self.y_tr[np.ix_(split_choice,np.ravel(t_C))])
            yTe = np.ravel(self.y_te[:, t_C])
            del scale, tmpXte

            results = blr.binary_log_regression(XTr, yTr, w_tr, XTe, yTe, out_dir)
            self.q2.put((t_C, i_C, i_T, i_R, sum(yTr), out_dir, eval_N, eval_type) + results)

            del t_C, i_C, i_R, i_T, eval_N, eval_type, split_choice, chosenFeatures
            del out_dir, w_tr, XTr, XTe, yTr, yTe
            gc.collect()

        self.q2.put("STOP")
        self.print_lock.acquire()
        print 'finished running classifiers at', time.time()
        self.print_lock.release()

    def make_table(self):
        records = []
        stops = 0
        while stops < self.nCPUs:
            record = self.q2.get()
            if record == 'STOP':
                stops += 1
            else:
                records.append(record)

        columns = ('class', 'iClass', 'iTrain', 'iRun', 'nTraining', 'out_dir', 
                'eval_n', 'eval_type', 'precision', 'recall', #  'support',
                'score', 'pr_auc', 'roc_auc', 'F', 'dprime') # , 'C')
        df = pandas.DataFrame.from_records(records, columns=columns)
        df.to_csv(self.out_file)
        del df

        self.print_lock.acquire()
        print 'finished making the table at', time.time()
        self.print_lock.release()

    def make_input(self, i_R, i_T, i_C):
        out_dir = os.path.join(self.eval_dir, str(i_C), str(i_T), str(i_R))
        data_file = mkdir(os.path.join(out_dir, 'data.mat'))
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

            # choose features
            if self.scores is None:
                tmpXtr = self.X_tr[split_choice,:]
                scale = skp.StandardScaler().fit(tmpXtr)
                XTr = scale.transform(tmpXtr)
                if self.permute:
                    perm = np.arange(XTr.shape[1])
                    np.random.shuffle(perm)
                    XTr = XTr[:,perm]
                chosenFeatures = utils.chooseFeatures(yTr,XTr,self.n_features,self.eval_type)
                del XTr, tmpXtr, scale
                gc.collect()
            else:
                scores = self.scores[split_choice,:]
                if self.permute:
                    perm = np.arange(scores.shape[1])
                    np.random.shuffle(perm)
                    scores = scores[:,perm]
                chosenFeatures = utils.chooseFeatures(yTr,scores,self.n_features,self.eval_type)
                del scores
                gc.collect()

            # save data
            scipy.io.savemat(data_file,{'class' : t_C,
                                        'iClass' : i_C,
                                        'nTraining' : sum(yTr),
                                        'iTrain' : i_T,
                                        'iRun' : i_R,
                                        'chosen_features' : chosenFeatures,
                                        'split_choice' : split_choice,
                                        'choices' : choices,
                                        'w_tr' : w_tr,
                                        'out_dir' : out_dir,
                                        'N' : self.eval_N,
                                        'type': self.eval_type})
        return data_file


if __name__ == '__main__':
    print 'started at', time.time()
    if len(sys.argv) < 4:
        tmp_dir = None
    else:
        tmp_dir = sys.argv[3]
    thing = PerformanceEvaluator(sys.argv[1], sys.argv[2], tmp_dir)
    print 'initialized at', time.time()
    thing.evaluatePerformance()
    print 'finished at', time.time()
