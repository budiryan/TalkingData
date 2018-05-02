import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import gc
import time
import datetime
import os
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import Imputer

# Data Constant definitions
DEBUG = False
NCHUNK = 120000000
OFFSET = 120000000
NROWS = 184903890
VAL_SIZE = 2500000
MISSING = -1
VERSION_NUM = '_v14_'
DTYPES = {
    'ip': 'uint32',
    'app': 'uint16',
    'device': 'uint16',
    'os': 'uint16',
    'channel': 'uint16',
    'is_attributed': 'uint8',
    'click_id': 'uint32',
}

# File directory constant definitions
DATAPATH = 'data/'
SUBMISSIONPATH = 'sub/'
PICKLEPATH = 'pickle/'
MODElPATH = 'model/'

# Performance stuff
NUM_CORES = 4 if DEBUG else 16

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
    gc.collect()
    return df, agg_name


def do_countuniq(df, group_cols, counted, agg_name, agg_type='int32', show_max=False, show_agg=True):
    if show_agg:
        print("Counting unqiue ", counted, " by ", group_cols, '...')
    gp = df[group_cols + [counted]].groupby(group_cols)[counted].nunique().reset_index().rename(
        columns={counted: agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print(agg_name + " max value = ", df[agg_name].max())
    df[agg_name] = df[agg_name].astype(agg_type)
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
    gc.collect()
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

    print('Imputing missing values ...')
    train_df['click_id'] = MISSING; train_df['click_id'] = train_df.click_id.astype('int8')
    len_train = len(train_df)
    test_df['is_attributed'] = MISSING; test_df['is_attributed'] = test_df.is_attributed.astype('int8')
    len_test = len(test_df)

    train_df = train_df.append(test_df)
    del test_df
    gc.collect()

    print('Extract day hour minute second from the dataset')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    train_df['minute'] = pd.to_datetime(train_df.click_time).dt.minute.astype('uint8')
    train_df['second'] = pd.to_datetime(train_df.click_time).dt.second.astype('uint8')
    gc.collect()

    predictors = ['app','device','os', 'channel', 'hour', 'day', 'minute', 'second']
    target = 'is_attributed'
    categorical = ['app', 'device', 'os', 'channel', 'hour', 'day', 'minute', 'second']

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
    train_df, new_feat = do_mean(train_df, ['app', 'channel'], 'hour', 'app_channel_mean_hour', show_max=True); gc.collect(); predictors.append(new_feat)
    train_df, new_feat = do_var(train_df, ['app', 'channel'], 'hour', 'app_channel_var_hour', show_max=True); gc.collect(); predictors.append(new_feat)
    print('Finished extracting new aggregated features')

    print('Generating new feature next_click and prev_click, also drop click time')
    train_df['click_time'] = (train_df['click_time'].astype(np.int64) // 10 ** 9).astype(np.int32)
    train_df['next_click'] = (train_df.groupby(['ip', 'app', 'device', 'os']).click_time.shift(-1) - train_df.click_time).astype(np.float32)
    train_df['prev_click'] = (train_df.click_time - train_df.groupby(['ip', 'app', 'device', 'os']).click_time.shift(1)).astype(np.float32)
    train_df.drop(['click_time'], axis=1, inplace=True)
    predictors.extend(['next_click', 'prev_click'])
    gc.collect()
    print('Finished generating next_click and prev_click')

    print('Generating mean encoding using random forrest')
    clf = RandomForestClassifier(n_jobs=-1)

    print('Training sklearn Random Forest')
    clf.fit(imputer.fit_transform(train_df[: len_train][predictors].values),
           train_df[: len_train][target].values)

    print('Predict using the trained random forest')
    train_df[len_train:]['is_attributed'] = clf.predict(imputer.fit_transform(train_df[len_train:][predictors].values))

    # print(train_df[len_train:][train_df[len_train:]['is_attributed'] > 0.0]['is_attributed'].shape)
    # print(np.count_nonzero(clf.predict(imputer.fit_transform(train_df[len_train:][predictors].values))))

    gc.collect()

    # Use super basic logistic regression to predict the current test set then using it to generate expanding mean
    # lr = LogisticRegression()
    # imputer = Imputer()
    #
    # print('Train using logistic regression to predict current test set')
    # lr.fit(imputer.fit_transform(train_df[: len_train][predictors].values),
    #        train_df[: len_train][target].values)
    # train_df[len_train: ]['is_attributed'] = lr.predict(imputer.fit_transform(train_df[len_train: ][predictors].values))
    # gc.collect()
    # max_mem_size_GB = 4 if DEBUG else 101
    # print('Predictors: ', predictors)
    # h2o.init(max_mem_size_GB=max_mem_size_GB, nthreads=NUM_CORES, ip="localhost", port=8888)
    # # Create classifier, balance classes
    # '''
    # preds = ['app', 'device', 'os', 'channel', 'hour', 'day', 'minute', 'second', 'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6',
    #  'X7', 'X8', 'ip_tcount', 'ip_app_count', 'ip_app_os_count', 'ip_tchan_count', 'ip_app_os_var',
    #  'ip_app_channel_var_day', 'ip_app_channel_mean_hour', 'ip_app_channel_var_hour', 'app_channel_mean_hour', 'app_channel_var_hour',
    #  'next_click', 'prev_click']
    # '''
    # h2o_dtypes = {'app': 'categorical',
    #               'device': 'categorical',
    #               'os': 'categorical',
    #               'channel': 'categorical',
    #               'hour': 'categorical',
    #               'day': 'categorical',
    #               'minute': 'categorical',
    #               'second': 'categorical',
    #               'X0': 'numeric',
    #               'X1': 'numeric',
    #               'X2': 'numeric',
    #               'X3': 'numeric',
    #               'X4': 'numeric',
    #               'X5': 'numeric',
    #               'X6': 'numeric',
    #               'X7': 'numeric',
    #               'X8': 'numeric',
    #               'ip_tcount': 'numeric',
    #               'ip_app_count': 'numeric',
    #               'ip_app_os_count': 'numeric',
    #               'ip_tchan_count': 'numeric',
    #               'ip_app_os_var': 'numeric',
    #               'ip_app_channel_var_day': 'numeric',
    #               'ip_app_channel_mean_hour': 'numeric',
    #               'ip_app_channel_var_hour': 'numeric',
    #               'app_channel_mean_hour': 'numeric',
    #               'app_channel_var_hour': 'numeric',
    #               'next_click': 'numeric',
    #               'prev_click': 'numeric',
    #               'is_attributed': 'categorical'}
    #
    # h2o_dtypes_test = {'app': 'categorical',
    #               'device': 'categorical',
    #               'os': 'categorical',
    #               'channel': 'categorical',
    #               'hour': 'categorical',
    #               'day': 'categorical',
    #               'minute': 'categorical',
    #               'second': 'categorical',
    #               'X0': 'numeric',
    #               'X1': 'numeric',
    #               'X2': 'numeric',
    #               'X3': 'numeric',
    #               'X4': 'numeric',
    #               'X5': 'numeric',
    #               'X6': 'numeric',
    #               'X7': 'numeric',
    #               'X8': 'numeric',
    #               'ip_tcount': 'numeric',
    #               'ip_app_count': 'numeric',
    #               'ip_app_os_count': 'numeric',
    #               'ip_tchan_count': 'numeric',
    #               'ip_app_os_var': 'numeric',
    #               'ip_app_channel_var_day': 'numeric',
    #               'ip_app_channel_mean_hour': 'numeric',
    #               'ip_app_channel_var_hour': 'numeric',
    #               'app_channel_mean_hour': 'numeric',
    #               'app_channel_var_hour': 'numeric',
    #               'next_click': 'numeric',
    #               'prev_click': 'numeric'}
    #
    # h2o_clf = h2o.estimators.random_forest.H2ORandomForestEstimator(balance_classes=True,
    #                                                                 stopping_metric='AUC')
    # h2o_columns = predictors + [target]
    # print('length of predictors are: ', len(predictors))
    # h2o_clf.train(predictors, target, training_frame=h2o.H2OFrame(train_df[: len_train][h2o_columns],
    #                                                               column_types=h2o_dtypes,
    #                                                               column_names=h2o_columns))
    #
    # h2o_train_perf = h2o_clf.model_performance(train=True)
    # print('Train performance: ', h2o_train_perf.auc())
    #
    # print('Begin predicting')
    # test_result = h2o_clf.predict(h2o.H2OFrame(train_df[len_train: ][predictors],
    #                                            column_types=h2o_dtypes_test,
    #                                            column_names=predictors)).asnumeric()
    #
    # print('test_result: ', (test_result['1.0'] > 0.5).as_data_frame(use_pandas=False, header=False))

    # test_result = np.array((test_result['0.0'] < test_result['1.0']).as_data_frame(use_pandas=False, header=False))\
    #     .ravel()\
    #     .astype(np.uint8)
    #
    # print('test_result is: ', test_result)
    # print('len test result: ', len(test_result))
    # print('number of ones: ', np.count_nonzero(test_result))

    gc.collect()

    print('Start generating expanding mean')
    cumsum = (train_df.groupby(['ip', 'app', 'device', 'os'])[target].cumsum() - train_df[target]).values
    cumcnt = (train_df.groupby(['ip', 'app', 'device', 'os'])[target].cumcount()).values
    train_df['feat_expanding_mean'] = cumsum / cumcnt
    train_df['feat_expanding_mean'].fillna(0, inplace=True)
    del cumsum, cumcnt
    gc.collect()
    predictors.append('feat_expanding_mean')
    print('Finished generating mean encoding')

    print('Finished feature engineering')
    finish = time.time()
    total_dur = finish - start
    print('The required time for feature engineering is: hour:minute:second = %s' % (datetime.timedelta(seconds=total_dur)))


    '''
    #######################
    #   BEFORE MODELLING  #
    #######################
    '''
    # divide the train into train, val and test:
    test_df = train_df[len_train:]
    val_df = train_df[(len_train - VAL_SIZE): len_train]
    train_df = train_df[:(len_train - VAL_SIZE)]
    gc.collect()
    print('Train shape: ', train_df.shape)
    print('Val shape: ', val_df.shape)
    print('Test shape: ', test_df.shape)

    print('Predictors are: ', predictors)
    print('Len predictors: ', len(predictors))
    print('Categorical: ', categorical)

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
    num_boost_round = 700
    categorical_features = categorical
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric': metrics,
        'learning_rate': 0.11,
        'num_leaves': 7,  # 2^max_depth - 1
        'max_depth': 3,  # -1 means no limit
        'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 100,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
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

    print("Re-reading test data...")
    test_df = pd.read_pickle(PICKLEPATH + 'test' + VERSION_NUM + '.pkl.gz')
    submission = pd.DataFrame()

    print("Predicting...")
    y_pred = bst.predict(test_df[predictors], num_iteration=bst.best_iteration)

    submission['click_id'] = test_df['click_id'].astype('uint32').values
    submission['is_attributed'] = y_pred

    print("Saving the prediction result...")
    submission.to_csv(SUBMISSIONPATH+ 'submission' + VERSION_NUM + '.csv', index=False, float_format='%.9f')

    print("Done")
    print(submission.head(3))
