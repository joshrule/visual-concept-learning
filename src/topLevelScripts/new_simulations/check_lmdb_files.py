import lmdb
import pandas as pd
import numpy as np
import caffe
import re
from scipy import misc

def check_a_csv_lmdb_combo(csvfile,lmdbfile):

    csvdata = pd.read_csv(csvfile)
    csvfiles = csvdata.file.values
    lmdbfiles = []

    count = 0

    env = lmdb.open(lmdbfile, readonly=True)

    nCSV = len(csvdata.index)
    nLMDB = env.stat()['entries']
    print 'length of csv: {:d}\nlength of lmdb: {:d}\nequal lengths: {}'.format(nCSV,nLMDB,nCSV==nLMDB)

    regexp = re.compile('/\S+\.JPEG')

    with env.begin() as txn:
        for key,raw_datum in txn.cursor():

            csvkey = regexp.search(key).group()
            lmdbfiles += [csvkey] if csvkey else []

            # for every 1000th image, check for label and data similarity
            if count % 1000 == 0 and csvkey:
                datum = caffe.proto.caffe_pb2.Datum()
                datum.ParseFromString(raw_datum)

                lmdbY = datum.label
                maybe_csvY = csvdata[csvdata.file == csvkey].label
                csvY = maybe_csvY.iloc[0] if len(maybe_csvY.index) == 1 else -1
                lmatch = lmdbY == csvY

                flat_x = np.fromstring(datum.data, dtype=np.uint8)
                x = flat_x.reshape(datum.channels, datum.height, datum.width)
                img = misc.imread(csvkey)
                if img.ndim == 2: # make images 3D if necessary
                    img = np.concatenate((img[np.newaxis,...],img[np.newaxis,...],img[np.newaxis,...]),axis=0)
                else:
                    img = np.rollaxis(img,2)
                img = img[::-1,:,:]
                dmatch = (x == img).all()

                if not (dmatch and lmatch):
                    print '{:d}: label match = {}, data match = {}'.format(count,lmatch,dmatch)
            count += 1

        if len(csvfiles) > len(lmdbfiles):
            print 'ERROR: csv has more entries - {:d} > {:d}'.format(len(csvfiles),len(lmdbfiles))
        elif len(csvfiles) < len(lmdbfiles):
            print 'ERROR: lmdb has more entries - {:d} < {:d}'.format(len(csvfiles),len(lmdbfiles))
        elif not np.in1d(csvfiles,lmdbfiles).all():
            print 'ERROR: lmdb missing csv entries'
        elif not np.in1d(lmdbfiles,csvfiles).all():
            print 'ERROR: csv missing lmdb entries'
        else:
            print 'entries are identical (save for order)'
    print

check_a_csv_lmdb_combo('/data1/josh/ruleRiesenhuber2013/tmp/tr_te.txt','/data2/image_sets/image_net/feat_val_lmdb')
check_a_csv_lmdb_combo('/data1/josh/ruleRiesenhuber2013/tmp/te_te.txt','/data2/image_sets/image_net/eval_val_lmdb')
check_a_csv_lmdb_combo('/data1/josh/ruleRiesenhuber2013/tmp/tr_tr.txt','/data2/image_sets/image_net/feat_train_lmdb')
check_a_csv_lmdb_combo('/data1/josh/ruleRiesenhuber2013/tmp/te_tr.txt','/data2/image_sets/image_net/eval_train_lmdb')
