import cPickle
import gzip
import numpy as np
import re
import pandas as pd

def process(inFilenames, outFilename):

    all_records = []

    for filename in inFilenames:
        # load the records
        with gzip.open(filename, 'rb') as fd:
            records = cPickle.load(fd)

        all_records += [new_record 
                        for record in records 
                        for new_record in records_from_record(record)]

    df = pd.DataFrame.from_records(all_records)
    df.to_csv(outFilename, index=False)


def records_from_record(record):
    regex = re.compile('^.+/googlenet-binary-([^/]+)/.+$')
    featureSet = re.search(regex, record['out_dir']).group(1)
    weights = np.ravel(record['model'].coef_).tolist()
    new_records = []
    for i, weight in enumerate(weights):
        new_record = {}
        new_record['feature_set'] = featureSet
        new_record['feature_id'] = i
        new_record['category'] = record['class']
        new_record['n_training'] = record['nTraining']
        new_record['split'] = record['iRun']
        new_record['weight'] = weight
        new_records.append(new_record)
    return new_records
