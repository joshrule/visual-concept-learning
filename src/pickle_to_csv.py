import cPickle
import gzip
import numpy as np
import re
import pandas as pd

def process(inFilenames, outFilename):
    '''Convert a gzipped, pickled, file of records to a CSV.

    The CSV reports the behavior of binary classifiers for the combined
    analysis. This function extracts one record per feature per classifier.
    '''
    all_records = []
    for filename in inFilenames:
        # Load the records.
        with gzip.open(filename, 'rb') as fd:
            records = cPickle.load(fd)

        # Add each new record to our list of records.
        all_records += [new_record 
                        for record in records 
                        for new_record in records_from_record(record)]

    # Create a dataframe and write it to CSV.
    df = pd.DataFrame.from_records(all_records)
    df.to_csv(outFilename, index=False)


def records_from_record(record):
    '''Split the record of a single binary classifier into one record per feature.'''
    # Identify the feature set.
    regex = re.compile('^.+/googlenet-binary-([^/]+)/.+$')
    featureSet = re.search(regex, record['out_dir']).group(1)
    # Compute the model weights.
    weights = np.ravel(record['model'].coef_).tolist()
    # Create and return the new records.
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
