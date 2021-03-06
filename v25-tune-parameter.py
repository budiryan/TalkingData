import pandas as pd
import numpy as np
import gc
import time
import datetime
import os
import lightgbm as lgb

from skopt.space import Real, Integer
from skopt.utils import use_named_args
import itertools
from sklearn.metrics import roc_auc_score
from skopt import gp_minimize
from skopt import dump

'''
Experimenting with skopt's bayesian hyperparameter tuning
'''

# Data Constant definitions
DEBUG = False

# Full dataset
# NCHUNK = 184903890
# OFFSET = 184903890

NCHUNK = 60000000
OFFSET = 60000000

NROWS = 184903890
VAL_SIZE = 2500000
MISSING = -1
VERSION_NUM = '_v25_'
DTYPES = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'int8',
    'click_id': 'int32',
}

# File directory constant definitions
DATAPATH = 'data/'
SUBMISSIONPATH = 'sub/'
PICKLEPATH = 'pickle/'
MODElPATH = 'model/'
FEATUREPATH = 'feature/'
SKOPTPATH = 'skopt/'

# Performance stuff
NUM_CORES = 4 if DEBUG else 8

# Various Feature Engineering Techniques
def do_count(df, group_cols, agg_name, agg_type='int32', show_max=False, show_agg=True):
    if show_agg:
        print("Aggregating by ", group_cols, '...')
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print(agg_name + " max value = ", df[agg_name].max())
    df[agg_name] = df[agg_name].astype(agg_type).fillna(0)
    gc.collect()
    return df, agg_name


def do_countuniq(df, group_cols, counted, agg_name, agg_type='int32', show_max=False, show_agg=True):
    if show_agg:
        print("Counting unique ", counted, " by ", group_cols, '...')
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].nunique().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print(agg_name + " max value = ", df[agg_name].max())
    df[agg_name] = df[agg_name].astype(agg_type).fillna(0)
    gc.collect()
    return df, agg_name


def do_cumcount(df, group_cols, counted, agg_name, agg_type='int32', show_max=False, show_agg=True):
    if show_agg:
        print("Cumulative count by ", group_cols, '...')
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name] = gp.values
    del gp
    if show_max:
        print(agg_name + " max value = ", df[agg_name].max())
    df[agg_name] = df[agg_name].astype(agg_type).fillna(0)
    gc.collect()
    return df, agg_name


def do_mean(df, group_cols, counted, agg_name, agg_type='float32', show_max=False, show_agg=True):
    if show_agg:
        print("Calculating mean of ", counted, " by ", group_cols, '...')
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].mean().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print(agg_name + " max value = ", df[agg_name].max())
    df[agg_name] = df[agg_name].astype(agg_type).fillna(0)
    gc.collect()
    return df, agg_name


def do_var(df, group_cols, counted, agg_name, agg_type='float32', show_max=False, show_agg=True):
    if show_agg:
        print("Calculating variance of ", counted, " by ", group_cols, '...')
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].var().reset_index().rename(columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print(agg_name + " max value = ", df[agg_name].max())
    df[agg_name] = df[agg_name].astype(agg_type).fillna(0)
    gc.collect()
    return df, agg_name


def do_next_click(df, group_cols, agg_name, agg_type='float32', show_min=False, show_agg=True):
    if show_agg:
        print('Calculating next click of ', group_cols)
    df[agg_name] = (df.groupby(group_cols).click_time.shift(-1)
                              .fillna(3000000000) - df.click_time).astype(agg_type)
    if show_min:
        print(agg_name + " mean value = ", df[agg_name].mean())
    return df, agg_name


def do_prev_click(df, group_cols, agg_name, agg_type='float32', show_min=False, show_agg=True):
    if show_agg:
        print('Calculating prev click of ', group_cols)
    df[agg_name] = (df.groupby(group_cols).click_time.shift(-1)
                              .fillna(3000000000) - df.click_time).astype(agg_type)
    if show_min:
        print(agg_name + " mean value = ", df[agg_name].mean())
    return df, agg_name


if __name__ == '__main__':
    '''
    ################
    # DATA IMPORTS #
    ################
    '''
    if DEBUG:
        print('Debug mode is on, load tiny subset of data set')
        train_df = pd.read_csv(DATAPATH + 'train_sample.csv',
                               parse_dates=['click_time'],
                               usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'])
        test_df = pd.read_csv(DATAPATH + 'test.csv', nrows=100000, parse_dates=['click_time'], dtype=DTYPES,
                              usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'])
        VAL_SIZE = 20000
    else:
        print('Debug mode is off, load the entire data set')
        train_df = pd.read_csv(DATAPATH + 'train.csv',
                               parse_dates=['click_time'],
                               skiprows=range(1, NROWS - OFFSET),
                               nrows=NCHUNK , dtype=DTYPES,
                               usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'])
        test_df = pd.read_csv(DATAPATH + 'test.csv', parse_dates=['click_time'], dtype=DTYPES,
                              usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id'])
    print('Finished data import...')


    '''
    #######################
    # FEATURE ENGINEERING #
    #######################
    '''
    start = time.time()
    print('Start feature engineering...')
    train_df['click_id'] = MISSING; train_df['click_id'] = train_df.click_id.astype('int32')
    len_train = len(train_df)
    gc.collect()

    test_df['is_attributed'] = MISSING; test_df['is_attributed'] = test_df.is_attributed.astype('int8')
    len_test = len(test_df)
    gc.collect()

    train_df = train_df.append(test_df)
    train_df = train_df.reset_index()
    del test_df
    gc.collect()

    print('Extract day hour minute second from the dataset')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    gc.collect()

    predictors = ['app','device','os', 'channel', 'hour', 'day']
    target = 'is_attributed'
    categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']

    print('Extracting aggregation features...')
    train_df, new_feat = do_countuniq(train_df, ['ip'], 'channel', 'X0', 'uint8', show_max=True);gc.collect(); predictors.append(new_feat)
    train_df, new_feat = do_cumcount(train_df, ['ip', 'device', 'os'], 'app', 'X1', show_max=True);gc.collect(); predictors.append(new_feat)
    train_df, new_feat = do_countuniq(train_df, ['ip', 'day'], 'hour', 'X2', 'uint8', show_max=True); gc.collect(); predictors.append(new_feat)
    train_df, new_feat = do_countuniq(train_df, ['ip'], 'app', 'X3', 'uint8', show_max=True);gc.collect(); predictors.append(new_feat)
    train_df, new_feat = do_countuniq(train_df, ['ip', 'app'], 'os', 'X4', 'uint8', show_max=True);gc.collect(); predictors.append(new_feat)
    train_df, new_feat = do_countuniq(train_df, ['ip'], 'device', 'X5', 'uint16', show_max=True); gc.collect(); predictors.append(new_feat)
    train_df, new_feat = do_countuniq(train_df, ['app'], 'channel', 'X6', show_max=True);gc.collect(); predictors.append(new_feat)
    train_df, new_feat = do_cumcount(train_df, ['ip'], 'os', 'X7', show_max=True);gc.collect(); predictors.append(new_feat)
    train_df, new_feat = do_countuniq(train_df, ['ip', 'device', 'os'], 'app', 'X8', show_max=True);gc.collect(); predictors.append(new_feat)
    train_df, new_feat = do_count(train_df, ['ip', 'day', 'hour'], 'ip_tcount', show_max=True);gc.collect(); predictors.append(new_feat)
    train_df, new_feat = do_count(train_df, ['ip', 'app'], 'ip_app_count', show_max=True);gc.collect(); predictors.append(new_feat)
    train_df, new_feat = do_count(train_df, ['ip', 'app', 'os'], 'ip_app_os_count', 'uint16', show_max=True);gc.collect(); predictors.append(new_feat)

    # device --> day
    train_df, new_feat = do_countuniq(train_df, ['device'], 'day', 'device_day_countuniq', 'uint32', True); gc.collect(); predictors.append(new_feat)

    # device, day --> hour
    train_df, new_feat = do_countuniq(train_df, ['device', 'day'], 'hour', 'device_day_hour_countuniq', 'uint32', True); gc.collect(); predictors.append(new_feat)

    # channel, day --> hour
    train_df, new_feat = do_countuniq(train_df, ['channel', 'day'], 'hour', 'channel_day_hour_countuniq', 'uint32', True); gc.collect(); predictors.append(new_feat)

    # ip, channel --> app
    train_df, new_feat = do_countuniq(train_df, ['ip', 'channel'], 'app', 'ip_channel_app_countuniq', 'uint32', True); gc.collect(); predictors.append(new_feat)

    # ip --> device
    train_df, new_feat = do_countuniq(train_df, ['ip'], 'device', 'ip_device_countuniq', 'uint32', True); gc.collect(); predictors.append(new_feat)
    print('Finished extracting new aggregated features')

    print('Generating new feature next_click and prev_click, also drop click time')
    train_df['click_time'] = (train_df['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)

    train_df, new_feat = do_next_click(train_df, ['ip', 'app', 'device', 'os'], 'next_click_0', 'int32', True); predictors.append(new_feat)
    train_df, new_feat = do_next_click(train_df, ['device'], 'next_click_1', 'int32', True); predictors.append(new_feat)
    train_df, new_feat = do_next_click(train_df, ['device', 'channel'], 'next_click_2', 'int32', True); predictors.append(new_feat)
    train_df, new_feat = do_next_click(train_df, ['app', 'device', 'channel'], 'next_click_3', 'int32', True); predictors.append(new_feat)
    train_df, new_feat = do_next_click(train_df, ['device', 'hour'], 'next_click_4', 'int32', True); predictors.append(new_feat)
    train_df, new_feat = do_next_click(train_df, ['ip', 'app', 'device', 'os', 'channel'], 'next_click_5', 'int32', True); predictors.append(new_feat)
    train_df, new_feat = do_next_click(train_df, ['ip', 'os', 'device'], 'next_click_6', 'int32', True); predictors.append(new_feat)

    # We do not do prev click
    train_df.drop(['click_time'], axis=1, inplace=True)
    gc.collect()
    print('Finished generating next_click and prev_click')

    '''
    #######################
    #   BEFORE MODELLING  #
    #######################
    '''
    # Remove day from the training set
    predictors.remove('day')
    categorical.remove('day')
    train_df.drop(['day'], axis=1, inplace=True)

    # divide the train into train, val and test:
    test_df = train_df[len_train:]
    val_df = train_df[(len_train - VAL_SIZE): len_train]
    train_df = train_df[:(len_train - VAL_SIZE)]
    gc.collect()
    print('Train shape: ', train_df.shape)
    print('Val shape: ', val_df.shape)
    print('Test shape: ', test_df.shape)

    print('Predictors are: ', predictors)
    print('Num of features is: ', len(predictors))
    print('Categorical variables: ', categorical)

    if not os.path.isfile(PICKLEPATH + 'train' + VERSION_NUM + '.pkl.gz'):
        print('Saving train file')
        train_df.to_pickle(PICKLEPATH + 'train' + VERSION_NUM + '.pkl.gz')
    else:
        print('Train file is there, no need to save')

    if not os.path.isfile(PICKLEPATH + 'val' + VERSION_NUM + '.pkl.gz'):
        print('Saving val file')
        val_df.to_pickle(PICKLEPATH + 'val' + VERSION_NUM + '.pkl.gz')
    else:
        print('Val file is there, no need to save')

    if not os.path.isfile(PICKLEPATH + 'test' + VERSION_NUM + '.pkl.gz'):
        print('Saving test file')
        test_df.to_pickle(PICKLEPATH + 'test' + VERSION_NUM + '.pkl.gz')
    else:
        print('Test file is there, no need to save')

    print('Save predictors and categorical: ')
    np.save(PICKLEPATH + 'predictors' + VERSION_NUM + '.npy', predictors)
    np.save(PICKLEPATH + 'categorical' + VERSION_NUM + '.npy', categorical)

    # Delete test df now to save memory
    del test_df
    gc.collect()


    '''
    #######################
    #   MODELLING         #
    #######################
    '''
    start_time = time.time()    
    space = [Integer(3, 10, name='max_depth'),
             Integer(6, 30, name='num_leaves'),
             Integer(20, 200, name='min_child_samples'),
             Real(250, 400,  name='scale_pos_weight'),
             Real(0.2, 0.7, name='subsample'),
             Real(0.2, 0.7, name='colsample_bytree'),
             Integer(50, 255, name='max_bin')
            ]

    print("Preparing train and val datasets")
    xgtrain = lgb.Dataset(train_df[predictors].values, label=train_df[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical,
                          free_raw_data=False
                          )
    del train_df
    gc.collect()


    xgvalid = lgb.Dataset(val_df[predictors].values, label=val_df[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical,
                          free_raw_data=False
                          )

    def objective(values):
        # Using LightGBM with Bayesian Parameter Tuning optimization
        early_stopping_rounds = 50
        verbose_eval = True
        num_boost_round = 1000

        lgb_params = {
            'max_depth': values[0],  # -1 means no limit
            'num_leaves': values[1],  # 2^max_depth - 1
            'min_child_samples': values[2],  # Minimum number of data need in a child(min_data_in_leaf)
            'scale_pos_weight': values[3],  # because training data is extremely unbalanced
            'subsample': values[4],  # Subsample ratio of the training instance.
            'colsample_bytree': values[5],  # Subsample ratio of columns when constructing each tree.
            'max_bin': values[6],  # Number of bucketed bin for feature values
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.1,
            'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
            'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
            'subsample_for_bin': 200000,  # Number of samples for constructing bin
            'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
            'reg_alpha': 0,  # L1 regularization term on weights
            'reg_lambda': 0,  # L2 regularization term on weights
            'nthread': NUM_CORES,
            'verbose': 0
        }

        evals_results = {}
        print('LGB PARAMETER: ', lgb_params)
        bst = lgb.train(lgb_params,
                        xgtrain,
                        valid_sets=[xgtrain, xgvalid],
                        valid_names=['train', 'valid'],
                        evals_result=evals_results,
                        num_boost_round=num_boost_round,
                        early_stopping_rounds=early_stopping_rounds,
                        verbose_eval=50,
                        feval=None)

        auc = -roc_auc_score(val_df[target], bst.predict(val_df[predictors], num_iteration=bst.best_iteration))

        print('\nAUROC.....',-auc,".....iter.....", bst.current_iteration())

        gc.collect()
        return auc

    res_gp = gp_minimize(objective, space, n_calls=20,
                     random_state=0, n_random_starts=10)

    print('Saving the optimization object: ')
    dump(res_gp, SKOPTPATH + 'skopt' + VERSION_NUM + '.gz')
    print('sk opt object saved')
    print('Best score= %.10f' % (res_gp.fun))
    print('minimum loc: ', res_gp.x)
    

    '''
    space = [Integer(3, 10, name='max_depth'),
             Integer(6, 30, name='num_leaves'),
             Integer(20, 200, name='min_child_samples'),
             Real(100, 400,  name='scale_pos_weight'),
             Real(0.2, 0.7, name='subsample'),
             Real(0.2, 0.7, name='colsample_bytree'),
             Integer(50, 255, name='max_bin')
            ]
    '''
    print("""Best parameters:
            - max_depth=%d
            - num_leaves=%d
            - min_child_samples=%d
            - scale_pos_weight=%d
            - subsample=%.4f
            - colsample_bytree=%.4f
            - max_bin=%.4f""" % (res_gp.x[0], res_gp.x[1], 
                                        res_gp.x[2], res_gp.x[3], 
                                        res_gp.x[4], res_gp.x[5],
                                        res_gp.x[6]))
