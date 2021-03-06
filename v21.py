import pandas as pd
import numpy as np
import gc
import time
import datetime
import os
import lightgbm as lgb

from sklearn.preprocessing import Imputer


'''
# MORE FEATURES LIKE V18, AND MORE DATASSSS 100M rows
'''

# Data Constant definitions
DEBUG = False
NCHUNK = 75000000
OFFSET = 75000000
NROWS = 184903890
VAL_SIZE = 2500000
MISSING = -1
VERSION_NUM = '_v21_'
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
    df[agg_name] = df[agg_name].astype(agg_type)
    imputer = Imputer()
    df[agg_name] = imputer.fit_transform(df[agg_name].values.reshape(-1, 1))
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
    df[agg_name] = df[agg_name].astype(agg_type)
    imputer = Imputer()
    df[agg_name] = imputer.fit_transform(df[agg_name].values.reshape(-1, 1))
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
    df[agg_name] = df[agg_name].astype(agg_type)
    imputer = Imputer()
    df[agg_name] = imputer.fit_transform(df[agg_name].values.reshape(-1, 1))
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
    df[agg_name] = df[agg_name].astype(agg_type)
    imputer = Imputer()
    df[agg_name] = imputer.fit_transform(df[agg_name].values.reshape(-1, 1))
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
    df[agg_name] = df[agg_name].astype(agg_type)
    imputer = Imputer()
    df[agg_name] = imputer.fit_transform(df[agg_name].values.reshape(-1, 1))
    gc.collect()
    return df, agg_name


def do_next_click(df, group_cols, agg_name, agg_type='float32', show_min=False, show_agg=True):
    if show_agg:
        print('Calculating next click of ', group_cols)
    df[agg_name] = (df.groupby(group_cols).click_time.shift(-1)
                              .fillna(3000000000) - df.click_time).astype(agg_type)
    if show_min:
        print(agg_name + " min value = ", df[agg_name].min())
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
    imputer = Imputer()
    print('Start feature engineering...')
    train_df['click_id'] = MISSING; train_df['click_id'] = train_df.click_id.astype('int32')

    # Only get day 9 as training set, see how it goes
    train_df = train_df[train_df.click_time.dt.day == 9].reset_index()
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
    train_df, new_feat = do_var(train_df, ['ip', 'day', 'channel'], 'hour', 'ip_tchan_count', show_max=True);gc.collect(); predictors.append(new_feat)
    train_df, new_feat = do_var(train_df, ['ip', 'app', 'os'], 'hour', 'ip_app_os_var', show_max=True);gc.collect(); predictors.append(new_feat)
    train_df, new_feat = do_var(train_df, ['ip', 'app', 'channel'], 'day', 'ip_app_channel_var_day', show_max=True);gc.collect(); predictors.append(new_feat)
    train_df, new_feat = do_mean(train_df, ['ip', 'app', 'channel'], 'hour', 'ip_app_channel_mean_hour', show_max=True);gc.collect(); predictors.append(new_feat)
    train_df, new_feat = do_var(train_df, ['ip', 'app', 'channel'], 'hour', 'ip_app_channel_var_hour', show_max=True);gc.collect(); predictors.append(new_feat)

    # device --> day
    train_df, new_feat = do_countuniq(train_df, ['device'], 'day', 'device_day_countuniq', 'uint32', True); gc.collect(); predictors.append(new_feat)
    train_df, new_feat = do_var(train_df, ['device'], 'day', 'device_day_var', show_max=True); gc.collect(); predictors.append(new_feat)

    # device, day --> hour
    train_df, new_feat = do_countuniq(train_df, ['device', 'day'], 'hour', 'device_day_hour_countuniq', 'uint32', True); gc.collect(); predictors.append(new_feat)
    train_df, new_feat = do_var(train_df, ['device', 'day'], 'hour', 'device_day_hour_var', show_max=True); gc.collect(); predictors.append(new_feat)

    # channel, day --> hour
    train_df, new_feat = do_countuniq(train_df, ['channel', 'day'], 'hour', 'channel_day_hour_countuniq', 'uint32', True); gc.collect(); predictors.append(new_feat)
    train_df, new_feat = do_var(train_df, ['channel', 'day'], 'hour', 'channel_day_hour_var', show_max=True); gc.collect(); predictors.append(new_feat)

    # ip, channel --> app
    train_df, new_feat = do_countuniq(train_df, ['ip', 'channel'], 'app', 'ip_channel_app_countuniq', 'uint32', True); gc.collect(); predictors.append(new_feat)

    # ip --> device
    train_df, new_feat = do_countuniq(train_df, ['ip'], 'device', 'ip_device_countuniq', 'uint32', True); gc.collect(); predictors.append(new_feat)
    print('Finished extracting new aggregated features')

    print('Generating new next click features, also drop click time')
    train_df['click_time'] = (train_df['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)

    train_df, new_feat = do_next_click(train_df, ['ip', 'app', 'device', 'os'], 'next_click_0', 'float32', True); predictors.append(new_feat)
    train_df, new_feat = do_next_click(train_df, ['device'], 'next_click_1', 'float32', True); predictors.append(new_feat)
    train_df, new_feat = do_next_click(train_df, ['device', 'channel'], 'next_click_2', 'float32', True); predictors.append(new_feat)
    train_df, new_feat = do_next_click(train_df, ['app', 'device', 'channel'], 'next_click_3', 'float32', True); predictors.append(new_feat)
    train_df, new_feat = do_next_click(train_df, ['device', 'hour'], 'next_click_4', 'float32', True); predictors.append(new_feat)

    # New
    train_df, new_feat = do_next_click(train_df, ['app', 'device', 'hour'], 'next_click_5', 'float32', True); predictors.append(new_feat)
    train_df, new_feat = do_next_click(train_df, ['channel', 'hour'], 'next_click_6', 'float32', True); predictors.append(new_feat)
    train_df, new_feat = do_next_click(train_df, ['app', 'channel', 'hour'], 'next_click_7', 'float32', True); predictors.append(new_feat)
    train_df, new_feat = do_next_click(train_df, ['device', 'channel', 'hour'], 'next_click_8', 'float32', True); predictors.append(new_feat)
    train_df, new_feat = do_next_click(train_df, ['app', 'device', 'channel', 'hour'], 'next_click_9', 'float32', True); predictors.append(new_feat)
    train_df, new_feat = do_next_click(train_df, ['device'], 'next_click_10', 'float32', True); predictors.append(new_feat)
    train_df, new_feat = do_next_click(train_df, ['device', 'hour'], 'next_click_11', 'float32', True); predictors.append(new_feat)
    train_df, new_feat = do_next_click(train_df, ['device', 'channel'], 'next_click_12', 'float32', True); predictors.append(new_feat)
    train_df, new_feat = do_next_click(train_df, ['app', 'device', 'channel'], 'next_click_13', 'float32', True); predictors.append(new_feat)
    train_df, new_feat = do_next_click(train_df, ['device', 'channel'], 'next_click_14', 'float32', True); predictors.append(new_feat)

    # We do not do prev click
    train_df.drop(['click_time'], axis=1, inplace=True)
    gc.collect()
    print('Finished generating next_click and prev_click')

    '''
    #######################
    #   BEFORE MODELLING  #
    #######################
    '''
    # Remove day from the training set because it does not make sense
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

    del test_df
    gc.collect()


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
    verbose_eval = True
    num_boost_round = 1500
    categorical_features = categorical
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

    print("Preparing validation datasets")
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
