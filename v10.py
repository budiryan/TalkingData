import pandas as pd
import time
import numpy as np
from sklearn.metrics import roc_auc_score
import catboost
import gc
import matplotlib.pyplot as plt
import os
import numpy as np


'''
catboost with larger dataset
'''


# Various Feature Engineering Measures
def do_count( df, group_cols, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Aggregating by ", group_cols , '...' )
    gp = df[group_cols][group_cols].groupby(group_cols).size().rename(agg_name).to_frame().reset_index()
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )

def do_countuniq( df, group_cols, counted, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Counting unqiue ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].nunique().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )
    
def do_cumcount( df, group_cols, counted, agg_name, agg_type='uint32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Cumulative count by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].cumcount()
    df[agg_name]=gp.values
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )

def do_mean( df, group_cols, counted, agg_name, agg_type='float32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Calculating mean of ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].mean().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )

def do_var( df, group_cols, counted, agg_name, agg_type='float32', show_max=False, show_agg=True ):
    if show_agg:
        print( "Calculating variance of ", counted, " by ", group_cols , '...' )
    gp = df[group_cols+[counted]].groupby(group_cols)[counted].var().reset_index().rename(columns={counted:agg_name})
    df = df.merge(gp, on=group_cols, how='left')
    del gp
    if show_max:
        print( agg_name + " max value = ", df[agg_name].max() )
    df[agg_name] = df[agg_name].astype(agg_type)
    gc.collect()
    return( df )

'''
DATA IMPORTS
'''
DEBUG = False
WHERE = 'gcloud'
NCHUNK = 40000000
OFFSET = 75000000

MISSING32 = 999999999
MISSING8 = 255
VERSION_NUM = '_v10_'

if WHERE=='kaggle':
    inpath = '../input/'
    pickle_path ='../input/pickle/'
    bin_path = '../input/bin/'
    cores = 4
elif WHERE=='gcloud':
    inpath = 'data/'
    pickle_path = 'pickle/'
    model_path = 'model/'
    bin_path = 'bin/'
    submission_path = 'sub/'
    cores = 8

debug = DEBUG
if debug:
    print('*** debug parameter set: this is a test run for debugging purposes ***')

nrows = 184903890
nchunk = NCHUNK
val_size = 2500000
frm = nrows - OFFSET

if debug:
    frm = 0
    nchunk = 100000
    val_size = 10000

to = frm + nchunk

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32',
        }

def get_df():
    predictors = []
    try:
        print('Try loading saved data...')
        train_df = pd.read_pickle(pickle_path + 'train' + VERSION_NUM + '.pkl.gz')
        val_df = pd.read_pickle(pickle_path + 'val' + VERSION_NUM + '.pkl.gz')
        test_df = pd.read_pickle(pickle_path + 'test' + VERSION_NUM + '.pkl.gz')
    except:
        print('Train val and test not there, redo feature engineering')
        train_df = pd.read_csv(inpath + "train.csv", parse_dates=['click_time'], skiprows=range(1,frm), nrows=to-frm, dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
        print('loading test data...')
        if debug:
            print('Debug mode activated, will load much fewer data')
            test_df = pd.read_csv(inpath + 'test.csv', nrows=100000, parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
        else:
            test_df = pd.read_csv(inpath + 'test.csv', parse_dates=['click_time'], dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
        train_df['click_id'] = MISSING32
        train_df['click_id'] = train_df.click_id.astype('uint32')
        print('finished loading data')

        print('Train shape: ', train_df.shape)
        print('Test shape: ', test_df.shape)

        len_train = len(train_df)
        test_df['is_attributed'] = MISSING8
        test_df['is_attributed'] = test_df.is_attributed.astype('uint8')
        train_df=train_df.append(test_df)


        del test_df
        gc.collect()

        '''
        FEATURE ENGINEERING
        '''

        # Extract hour of the day
        print('Extracting hour as a new feature...')
        train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
        train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
        print('Finished')

        print('Extracting aggregation features...')
        train_df = do_cumcount( train_df, ['ip', 'device', 'os'], 'app', 'X1', show_max=True ); gc.collect()
        train_df = do_cumcount( train_df, ['ip'], 'os', 'X7', show_max=True ); gc.collect()
        train_df = do_countuniq( train_df, ['ip'], 'channel', 'X0', 'uint8', show_max=True ); gc.collect()
        train_df = do_countuniq( train_df, ['ip', 'day'], 'hour', 'X2', 'uint8', show_max=True ); gc.collect()
        train_df = do_countuniq( train_df, ['ip'], 'app', 'X3', 'uint8', show_max=True ); gc.collect()
        train_df = do_countuniq( train_df, ['ip', 'app'], 'os', 'X4', 'uint8', show_max=True ); gc.collect()
        train_df = do_countuniq( train_df, ['ip'], 'device', 'X5', 'uint16', show_max=True ); gc.collect()
        train_df = do_countuniq( train_df, ['app'], 'channel', 'X6', show_max=True ); gc.collect()
        train_df = do_countuniq( train_df, ['ip', 'device', 'os'], 'app', 'X8', show_max=True ); gc.collect()
        train_df = do_count( train_df, ['ip', 'day', 'hour'], 'ip_tcount', show_max=True ); gc.collect()
        train_df = do_count( train_df, ['ip', 'app'], 'ip_app_count', show_max=True ); gc.collect()
        train_df = do_count( train_df, ['ip', 'app', 'os'], 'ip_app_os_count', 'uint16', show_max=True ); gc.collect()
        train_df = do_var( train_df, ['ip', 'day', 'channel'], 'hour', 'ip_tchan_count', show_max=True ); gc.collect()
        train_df = do_var( train_df, ['ip', 'app', 'os'], 'hour', 'ip_app_os_var', show_max=True ); gc.collect()
        train_df = do_var( train_df, ['ip', 'app', 'channel'], 'day', 'ip_app_channel_var_day', show_max=True ); gc.collect()
        train_df = do_mean( train_df, ['ip', 'app', 'channel'], 'hour', 'ip_app_channel_mean_hour', show_max=True ); gc.collect()
        print('Finished extracting new features')

        print('Doing nextClick...')
        new_feature = 'nextClick'
        D = 2 ** 26
        train_df['category'] = (train_df['ip'].astype(str) + "_" + train_df['app'].astype(str) + "_" + train_df['device'].astype(str) \
                + "_" + train_df['os'].astype(str)).apply(hash) % D
        click_buffer= np.full(D, 3000000000, dtype=np.uint32)
        train_df['epochtime']= train_df['click_time'].astype(np.int64) // 10 ** 9
        next_clicks = []
        for category, t in zip(reversed(train_df['category'].values), reversed(train_df['epochtime'].values)):
            next_clicks.append(click_buffer[category]-t)
            click_buffer[category] = t
        del(click_buffer)
        QQ = np.log2(1 + np.array(list(reversed(next_clicks))))
        train_df[new_feature] = pd.Series(QQ).astype('float32')
        del QQ, next_clicks
        gc.collect()
        print('Finished doing nextClick...')

        print('Doing prevClick...')
        new_feature = 'prevClick'
        click_buffer= np.zeros(D, dtype=np.uint32)
        prev_clicks= []

        for category, t in zip(train_df['category'].values, train_df['epochtime'].values):
            prev_clicks.append(t - click_buffer[category])
            click_buffer[category] = t

        del(click_buffer)
        QQ = np.log2(1 + np.array(list(prev_clicks)))
        train_df.drop(['epochtime','category','click_time'], axis=1, inplace=True)
        train_df[new_feature] = pd.Series(QQ).astype('float32')
        del QQ, prev_clicks
        gc.collect()
        print('Finished doing prevClick...')

        # Set all the predictor features and target
        print("vars and data type: ")
        train_df['ip_tcount'] = train_df['ip_tcount'].astype('uint16')
        train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
        train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')

        '''
        MODELLING
        '''

        # Split into train,test and val
        test_df = train_df[len_train: ]
        val_df = train_df[(len_train-val_size): len_train]
        train_df = train_df[:(len_train - val_size)]
        train_df.to_pickle(pickle_path + 'train' + VERSION_NUM + '.pkl.gz')
        val_df.to_pickle(pickle_path + 'val' + VERSION_NUM + '.pkl.gz')
    target = 'is_attributed'
    predictors.extend(['app','device','os', 'channel', 'hour', 'day',
                  'nextClick', 'prevClick', 'ip_tcount', 
                  'ip_app_count', 'ip_tchan_count',
                  'ip_app_os_count', 'ip_app_os_var',
                  'ip_app_channel_var_day','ip_app_channel_mean_hour',
                  'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8'])
    categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']
    return test_df, val_df, train_df, predictors, categorical, target


test_df, val_df, train_df, predictors, categorical, target = get_df()

# |TRAIN -- VAL -- TEST|
print("train size: ", len(train_df))
print("valid size: ", len(val_df))
print("test size : ", len(test_df))

# Save little bit of memory
print('Save the dataframes')
if not os.path.isdir(pickle_path):
    os.makedirs(pickle_path)
test_df.to_pickle(pickle_path + 'test' + VERSION_NUM + '.pkl.gz')

del test_df
gc.collect()

# Training part
start_time = time.time()

# Use catboost here
# df['predictors'], df['target']
'''
Categorical info:
0      74823373
255    18790469
1        176627
'''
# Get categorical features indices
print('categorical is: ', categorical)
cat_indices = [train_df[predictors].columns.get_loc(c) for c in categorical]
print('cat indices are: ', cat_indices)
print(train_df[predictors].columns)

# Create some pools for train + val
print("Create pool object")
train_pool = catboost.Pool(train_df[predictors], train_df[target], cat_indices)
val_pool = catboost.Pool(val_df[predictors], val_df[target], cat_indices)
cols = train_df.columns
del train_df, val_df
gc.collect()

cb = catboost.CatBoostClassifier(iterations=1000, 
                        learning_rate=0.01, 
                        depth=7, 
                        loss_function='Logloss', 
                        eval_metric='AUC', 
                        random_seed=99, 
                        od_type='Iter', 
                        scale_pos_weight=400,
                        l2_leaf_reg=0,
                        od_wait=50,
                        thread_count=8,
                        max_ctr_complexity=1,
                        used_ram_limit=28e10,
                        ) 

print("Begin fitting to model")
cb.fit(X=train_pool, 
       eval_set=val_pool,
       use_best_model=True, 
       verbose=True)

print('Feature importance are: ')
imp = cb.get_feature_importance(train_pool)
print(imp)

print("Re-reading test data...")
test_df = pd.read_pickle(pickle_path + 'test' + VERSION_NUM + '.pkl.gz')
submission = pd.DataFrame()


print("Predicting...")
# Predict using catboost here
print('predictors here are: ', predictors)
y_pred = np.array(cb.predict_proba(test_df[predictors]))[:, 1]
print('y_pred is: ', y_pred)

submission['click_id'] = test_df['click_id'].astype('uint32').values
submission['is_attributed'] = y_pred

print("Saving the prediction result...")
if not os.path.isdir(submission_path):
    os.makedirs(submission_path)
submission.to_csv(submission_path + 'submission' + VERSION_NUM +'.csv', index=False, float_format='%.9f')

print("Done, sample submission: ")
print(submission.head(10))
