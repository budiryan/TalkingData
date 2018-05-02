import lightgbm as lgb
import pandas as pd 


# File directory constant definitions
DATAPATH = 'data/'
SUBMISSIONPATH = 'sub/'
PICKLEPATH = 'pickle/'
MODElPATH = 'model/'
FEATUREPATH = 'feature/'
VERSION_NUM = '_v18_'

predictors = ['app', 'device', 'os', 'channel', 'hour', 'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'ip_tcount', 
              'ip_app_count', 'ip_app_os_count', 'ip_tchan_count', 'ip_app_os_var', 'ip_app_channel_var_day', 'ip_app_channel_mean_hour', 
              'ip_app_channel_var_hour', 'device_day_countuniq', 'device_day_var', 'device_day_hour_countuniq', 'device_day_hour_var', 
              'channel_day_hour_countuniq', 'channel_day_hour_var', 'ip_channel_app_countuniq', 'ip_device_countuniq', 'next_click_0',
              'next_click_1', 'next_click_2', 'next_click_3', 'next_click_4']

print('len predictors: ', len(predictors))

print('Read test data')
test_df = pd.read_pickle(PICKLEPATH + 'test' + VERSION_NUM + '.pkl.gz')

print('Read saved BST')
bst = lgb.Booster(model_file=MODElPATH + 'bst' + VERSION_NUM + '.txt')  #init model

# read feature importance
print('FEATURE IMPORTANCE: ')
gain = bst.feature_importance('gain')
ft = pd.DataFrame({'feature': bst.feature_name(), 'split': bst.feature_importance('split'),
                   'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
ft.to_csv(FEATUREPATH + VERSION_NUM + '.csv')
print(ft.head(25))


# Predict
print('Predicting')
submission = pd.DataFrame()
y_pred = bst.predict(test_df[predictors], num_iteration=bst.best_iteration)
submission['click_id'] = test_df['click_id'].astype('uint32').values
submission['is_attributed'] = y_pred

print('Saving the prediction result...')
submission.to_csv(SUBMISSIONPATH+ 'submission' + VERSION_NUM + '.csv', index=False, float_format='%.9f')

print('Done')
print(submission.head(3))