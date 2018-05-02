import pandas as pd 
import lightgbm as lgb
import numpy as np 
import time
import gc
import datetime


'''
Using full datasets, 35 features, modify LGB parameters to subsample only 0.3 of the data
'''

DEBUG = False

# File directory constant definitions
VERSION_NUM = '_v23_'
VERSION_PREV = '_v22_'
DATAPATH = 'data/'
SUBMISSIONPATH = 'sub/'
PICKLEPATH = 'pickle/'
MODElPATH = 'model/'
FEATUREPATH = 'feature/'

'''
#######################
#      LOAD DATA      #
#######################
'''
predictors = list(np.load(PICKLEPATH + 'predictors' + VERSION_PREV + '.npy'))
categorical = list(np.load(PICKLEPATH + 'categorical' + VERSION_PREV + '.npy'))
target = 'is_attributed'

print('predictors: ', predictors)
print('categorical: ', categorical)


'''
#######################
#   MODELLING  #
#######################
'''
# Using LightGBM
start_time = time.time()

# Load model here
bst = lgb.Booster(model_file='model/' + 'bst' + VERSION_NUM + '.txt')

print('Re-reading test data...')
test_df = pd.read_pickle(PICKLEPATH + 'test' + VERSION_PREV + '.pkl.gz')
submission = pd.DataFrame()

print('Predicting...')
y_pred = bst.predict(test_df[predictors], num_iteration=bst.best_iteration)

print('FEATURE IMPORTANCE: ')
gain = bst.feature_importance('gain')
ft = pd.DataFrame({'feature': bst.feature_name(), 'split': bst.feature_importance('split'),
                   'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
ft.to_csv(FEATUREPATH + VERSION_NUM + '.csv')
print(ft)

submission['click_id'] = test_df['click_id'].astype('uint32').values
submission['is_attributed'] = y_pred

print('Saving the prediction result...')
submission.to_csv(SUBMISSIONPATH+ 'submission' + VERSION_NUM + '.csv', index=False, float_format='%.9f')

print('Done')
print(submission.head(3))
