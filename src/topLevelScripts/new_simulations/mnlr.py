import numpy as np
import scipy.io
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import sys

# get the file name from the arguments
thefilename = sys.argv[1]

# load the file
mat = scipy.io.loadmat(thefilename)

print 'data loaded %s' % str(datetime.now())
from datetime import datetime
# setup and fit the classifier
model = LogisticRegression(solver='sag', 
                           max_iter=100,
                           random_state=1,
                           n_jobs=16,
                           multi_class='multinomial').fit(mat['trainX'],np.ravel(mat['trainY']))
print 'model fit %s' % str(datetime.now())

# make the predictions
classificationValues = model.predict_proba(mat['testX'])
print 'predictions made %s' % str(datetime.now())
score1 = model.score(mat['trainX'],np.ravel(mat['trainY']))
score2 = model.score(mat['testX'],np.ravel(mat['testY']))
print 'scores computed %s' % str(datetime.now())
print 'training score : %.3f' % score1
print 'testing score : %.3f'  % score2

# save the predictions and classifier into the file
scipy.io.savemat(thefilename,{'model' : model, 'classificationValues' : classificationValues})
print 'all done %s' % str(datetime.now())
