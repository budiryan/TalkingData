import pandas as pd 
import lightgbm as lgb
import numpy as np 
import time
import gc
import datetime


DEBUG = False

# File directory constant definitions
VERSION_NUM = '_v22_'
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
# Load train data and val data
print('Loading train data and val data')

# Performance stuff
NUM_CORES = 4 if DEBUG else 16
train_df = pd.read_pickle(PICKLEPATH + 'train' + VERSION_NUM + '.pkl.gz')
val_df = pd.read_pickle(PICKLEPATH + 'val' + VERSION_NUM + '.pkl.gz')
predictors = list(np.load(PICKLEPATH + 'predictors' + VERSION_NUM + '.npy'))
categorical = list(np.load(PICKLEPATH + 'categorical' + VERSION_NUM + '.npy'))
target = 'is_attributed'

print('train df: ', train_df[predictors].shape)
print('val df: ', val_df[predictors].shape)
print('predictors: ', len(predictors))
print('categorical: ', len(categorical))


'''
#######################
#   MODELLING  #
#######################
'''
# Using LightGBM
start_time = time.time()
objective = 'binary'
metrics = 'auc'
early_stopping_rounds = 50
num_boost_round = 1500
verbose_eval = True
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': objective,
    'metric': metrics,
    'learning_rate': 0.1,
    'num_leaves': 7,  # 2^max_depth - 1
    'max_depth': 3,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': 0.5,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.5,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight': 400,  # because training data is extremely unbalanced
    'subsample_for_bin': 200000,  # Number of samples for constructing bin
    'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
    'reg_alpha': 0,  # L1 regularization term on weights
    'reg_lambda': 0,  # L2 regularization term on weights
    'nthread': NUM_CORES,
    'verbose': 0,
}

print("Preparing training and validation datasets")
xgtrain = lgb.Dataset(train_df[predictors].values, label=train_df[target].values,
                      feature_name=predictors,
                      categorical_feature=categorical
                      )
del train_df
gc.collect()


xgvalid = lgb.Dataset(val_df[predictors].values, label=val_df[target].values,
                      feature_name=predictors,
                      categorical_feature=categorical
                      )
del val_df
gc.collect()

evals_results = {}
print('LGB PARAMETER: ', lgb_params)
bst = lgb.train(lgb_params,
                xgtrain,
                valid_sets=[xgtrain, xgvalid],
                valid_names=['train', 'valid'],
                evals_result=evals_results,
                num_boost_round=num_boost_round,
                early_stopping_rounds=early_stopping_rounds,
                verbose_eval=10,
                feval=None)

print("\nModel Report")
print("bst.best_iteration: ", bst.best_iteration)
print(metrics + ":", evals_results['valid'][metrics][bst.best_iteration - 1])

print('The required time for model training is: hour:minute:second = %s' % (
    datetime.timedelta(seconds=time.time() - start_time)))

print('Saving the trained lightgbm model')
bst.save_model(MODElPATH + 'bst' + VERSION_NUM + '.txt')

print('Re-reading test data...')
test_df = pd.read_pickle(PICKLEPATH + 'test' + VERSION_NUM + '.pkl.gz')
submission = pd.DataFrame()

print('Predicting...')
y_pred = bst.predict(test_df[predictors], num_iteration=bst.best_iteration)

print('FEATURE IMPORTANCE: ')
gain = bst.feature_importance('gain')
ft = pd.DataFrame({'feature': bst.feature_name(), 'split': bst.feature_importance('split'),
                   'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
ft.to_csv(FEATUREPATH + VERSION_NUM + '.csv')
print(ft.head(25))

submission['click_id'] = test_df['click_id'].astype('uint32').values
submission['is_attributed'] = y_pred

print('Saving the prediction result...')
submission.to_csv(SUBMISSIONPATH+ 'submission' + VERSION_NUM + '.csv', index=False, float_format='%.9f')

print('Done')
print(submission.head(3))
