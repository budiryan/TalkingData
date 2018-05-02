import pandas as pd
import time
import numpy as np
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import gc
import matplotlib.pyplot as plt
import os


'''
Remove all features which are day related
'''

'''
DATA IMPORTS
'''
DEBUG = False
WHERE = 'gcloud'
NCHUNK = 32000000
OFFSET = 75000000

MISSING32 = 999999999
MISSING8 = 255
VERSION_NUM = '_v3_'

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
    cores = 7

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


print('loading train data...',frm,to)
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
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
print('Finished')


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

print('Extracting aggregation features...')
train_df = do_cumcount( train_df, ['ip', 'device', 'os'], 'app', 'X1', show_max=True ); gc.collect()
train_df = do_cumcount( train_df, ['ip'], 'os', 'X7', show_max=True ); gc.collect()
train_df = do_countuniq( train_df, ['ip'], 'channel', 'X0', 'uint8', show_max=True ); gc.collect()
train_df = do_countuniq( train_df, ['ip'], 'app', 'X3', 'uint8', show_max=True ); gc.collect()
train_df = do_countuniq( train_df, ['ip', 'app'], 'os', 'X4', 'uint8', show_max=True ); gc.collect()
train_df = do_countuniq( train_df, ['ip'], 'device', 'X5', 'uint16', show_max=True ); gc.collect()
train_df = do_countuniq( train_df, ['app'], 'channel', 'X6', show_max=True ); gc.collect()
train_df = do_countuniq( train_df, ['ip', 'device', 'os'], 'app', 'X8', show_max=True ); gc.collect()
train_df = do_count( train_df, ['ip', 'app'], 'ip_app_count', show_max=True ); gc.collect()
train_df = do_count( train_df, ['ip', 'app', 'os'], 'ip_app_os_count', 'uint16', show_max=True ); gc.collect()
train_df = do_var( train_df, ['ip', 'app', 'os'], 'hour', 'ip_app_os_var', show_max=True ); gc.collect()
train_df = do_mean( train_df, ['ip', 'app', 'channel'], 'hour', 'ip_app_channel_mean_hour', show_max=True ); gc.collect()
print('Finished extracting new features')

print('Doing nextClick...')
predictors=[]
new_feature = 'nextClick'
D=2**26
train_df['category'] = (train_df['ip'].astype(str) + "_" + train_df['app'].astype(str) + "_" + train_df['device'].astype(str) \
        + "_" + train_df['os'].astype(str)).apply(hash) % D
click_buffer= np.full(D, 3000000000, dtype=np.uint32)
train_df['epochtime']= train_df['click_time'].astype(np.int64) // 10 ** 9
next_clicks= []
for category, t in zip(reversed(train_df['category'].values), reversed(train_df['epochtime'].values)):
    next_clicks.append(click_buffer[category]-t)
    click_buffer[category]= t
del(click_buffer)
QQ= list(reversed(next_clicks))
train_df.drop(['epochtime','category','click_time'], axis=1, inplace=True)
train_df[new_feature] = pd.Series(QQ).astype('float32')
predictors.append(new_feature)
train_df[new_feature+'_shift'] = train_df[new_feature].shift(+1).values
predictors.append(new_feature+'_shift')
del QQ, next_clicks
gc.collect()
print('Finished doing nextClick...')

# Set all the predictor features and target
print("vars and data type: ")
train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')

target = 'is_attributed'
predictors.extend(['app','device','os', 'channel', 'hour',
                'ip_app_count', 'ip_app_os_count', 'ip_app_os_var',
              'ip_app_channel_mean_hour',
              'X0', 'X1', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8'])
categorical = ['app', 'device', 'os', 'channel', 'hour']
print('predictors',predictors)

'''
MODELLING
'''

# Split into train,test and val
test_df = train_df[len_train: ]
val_df = train_df[(len_train-val_size): len_train]
train_df = train_df[:(len_train - val_size)]

# |TRAIN -- VAL -- TEST|
print("train size: ", len(train_df))
print("valid size: ", len(val_df))
print("test size : ", len(test_df))

# Save little bit of memory
if not os.path.isdir(pickle_path):
    os.makedirs(pickle_path)
test_df.to_pickle(pickle_path + 'test' + VERSION_NUM + '.pkl.gz')

del test_df
gc.collect()

# Training part
start_time = time.time()
objective='binary' 
metrics='auc'
early_stopping_rounds=30 
verbose_eval=True 
num_boost_round=1000
categorical_features=categorical
lgb_params = {
    'boosting_type': 'gbdt',
    'objective': objective,
    'metric':metrics,
    'learning_rate': 0.10,
    'num_leaves': 7,  # 2^max_depth - 1
    'max_depth': 3,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': 0.7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight':200, # because training data is extremely unbalanced 
    'subsample_for_bin': 200000,  # Number of samples for constructing bin
    'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
    'reg_alpha': 0,  # L1 regularization term on weights
    'reg_lambda': 0,  # L2 regularization term on weights
    'nthread': cores,
    'verbose': 0,
    'metric':metrics
}

print("preparing validation datasets")
xgtrain = lgb.Dataset(train_df[predictors].values, label=train_df[target].values,
                      feature_name=predictors,
                      categorical_feature=categorical
                      )
                      
print( train_df[predictors].head() )
print( train_df[target].head() )
print("train df tail: ")
print( train_df[target].tail())
print( val_df[predictors].head() )
print( val_df[target].head() )
del train_df

if WHERE != 'kaggle':
    if not os.path.isdir(bin_path):
        os.makedirs(bin_path)
    xgtrain.save_binary(bin_path + 'xgtrain' + VERSION_NUM + '.bin')
    del xgtrain

xgvalid = lgb.Dataset(val_df[predictors].values, label=val_df[target].values,
                      feature_name=predictors,
                      categorical_feature=categorical
                      )

del val_df
gc.collect()


evals_results = {}

if WHERE != 'kaggle':
    xgtrain = lgb.Dataset(bin_path + 'xgtrain' + VERSION_NUM + '.bin',
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
                      
print('lgb_params: ', lgb_params )


bst = lgb.train(lgb_params, 
                 xgtrain, 
                 valid_sets=[xgtrain, xgvalid], 
                 valid_names=['train','valid'], 
                 evals_result=evals_results, 
                 num_boost_round=num_boost_round,
                 early_stopping_rounds=early_stopping_rounds,
                 verbose_eval=10, 
                 feval=None)

print("\nModel Report")
print("bst.best_iteration: ", bst.best_iteration)
print(metrics+":", evals_results['valid'][metrics][bst.best_iteration-1])


print('[{}]: model training time'.format(time.time() - start_time))

print('Saving the trained lightgbm model')
if not os.path.isdir(model_path):
    os.makedirs(model_path)
if not os.path.exists(model_path + 'bst' + VERSION_NUM + '.txt'):
    bst.save_model(model_path + 'bst' + VERSION_NUM + '.txt')

if WHERE!='gcloud':
    print('Plot feature importances...')
    ax = lgb.plot_importance(bst, max_num_features=100)
    plt.show()


print("Re-reading test data...")
test_df = pd.read_pickle(pickle_path + 'test' + VERSION_NUM + '.pkl.gz')
submission = pd.DataFrame()


print("Predicting...")
y_pred = bst.predict(test_df[predictors], num_iteration=bst.best_iteration)

submission['click_id'] = test_df['click_id'].astype('uint32').values
submission['is_attributed'] = y_pred

print("Saving the prediction result...")
if not os.path.isdir(submission_path):
    os.makedirs(submission_path)
submission.to_csv(submission_path + 'submission' + VERSION_NUM +'.csv', index=False, float_format='%.9f')

print("Done")
print(submission.head(3))
