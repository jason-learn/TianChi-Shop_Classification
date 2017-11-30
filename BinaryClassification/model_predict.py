# *coding=utf-8*
import gc
import pandas as pd
import numpy as np
from utils import bearing_array,hafuman_km,get_features_list



features = get_features_list()

dir = '../input/'
print('reading train/val data ')
train = pd.read_csv(dir + 'offline_train.csv')
val = pd.read_csv(dir + 'offline_val.csv')


shop_info_tmp = pd.read_csv(dir + 'ccf_first_round_shop_info.csv')
shop_info = shop_info_tmp[['shop_id','category_id','longitude','latitude','price']]
del shop_info_tmp;gc.collect()
shop_info.rename(columns={'longitude':'s_longitude','latitude':'s_latitude'},inplace=True)

train.rename(columns={'shop_id_x':'shop_id'},inplace=True)
val.rename(columns={'shop_id_x':'shop_id'},inplace=True)

user_info_tmp = pd.read_csv(dir + 'ccf_first_round_user_shop_behavior.csv').reset_index()
user_info = user_info_tmp[['index','longitude','latitude']]
del user_info_tmp;gc.collect()

train = pd.merge(train,shop_info,on=['shop_id'],how='left')
train = train.dropna()
train = pd.merge(train,user_info,on=['index'],how='left')

train['time_stamp'] = pd.to_datetime(train['time_stamp'])
train['current_hour'] =  pd.DatetimeIndex(train.time_stamp).hour
# train['current_week'] =  pd.DatetimeIndex(train.time_stamp).dayofweek

train['distance'] = hafuman_km(train['s_longitude'],train['s_latitude'],train['longitude'],train['latitude'])
train['current_bearing_array'] = bearing_array(train.s_latitude.values, train.s_longitude.values,
                                                                                    train.latitude.values, train.longitude.values)
# train['distance'] = np.log1p(train['distance'])

# 由于历史记录的意义并没有那么明确，因此将历史转化为和当前的差值处理
train['c_wifi_var'] = train['strength'] - train['c_sw_average']
train['wifi_var'] = train['strength'] - train['s_avg_power']

train['s_wifi_var'] = train['strength'] - train['w_avg_power']
train['sb_wifi_var'] = train['strength'] - train['sb_history_avg_power']


train['angle_var'] = train['history_bearing_array_median'] - train['current_bearing_array']
train['c_sb_wifi_var'] = train['strength'] - train['c_sb_history_avg_power']

train['s_sb_wifi_var_ratio'] = train['c_sb_wifi_var'] / (train['sb_wifi_var']+0.0001)
train['category_id'] = train['category_id'].map(lambda x:str(x).split('_')[1])
train['mall_id'] = train['mall_id'].map(lambda x:str(x).split('_')[1])

# train['sw_average_ratio'] = (train['c_sw_average'] + 0.5 ) / (train['sw_average'] + 1)
# train['bw_average_ratio'] = (train['c_bw_average'] + 0.5 )/ (train['bw_average'] + 1)

# train['price'] = np.log1p(train['price'])


# 特征字段
train_train_label = train.pop('label')
y_train = train_train_label.values
X_train = train[features].values
del train;gc.collect()
del train_train_label;gc.collect()
gc.collect()


val = pd.merge(val,shop_info,on=['shop_id'],how='left')
val = val.dropna()
val = pd.merge(val,user_info,on=['index'],how='left')

val['time_stamp'] = pd.to_datetime(val['time_stamp'])
val['current_hour'] =  pd.DatetimeIndex(val.time_stamp).hour
# val['current_week'] =  pd.DatetimeIndex(val.time_stamp).dayofweek

del user_info;gc.collect()
val['distance'] = hafuman_km(val['s_longitude'],val['s_latitude'],val['longitude'],val['latitude'])
val['current_bearing_array'] = bearing_array(val.s_latitude.values, val.s_longitude.values,
                                                                                    val.latitude.values, val.longitude.values)
#val['distance'] = np.log1p(val['distance'])

val['c_wifi_var'] = val['strength'] - val['c_sw_average']
val['wifi_var'] = val['strength'] - val['s_avg_power']
val['s_wifi_var'] = val['strength'] - val['w_avg_power']
val['sb_wifi_var'] = val['strength'] - val['sb_history_avg_power']
val['angle_var'] = val['history_bearing_array_median'] - val['current_bearing_array']
val['c_sb_wifi_var'] = val['strength'] - val['c_sb_history_avg_power']

val['s_sb_wifi_var_ratio'] = val['c_sb_wifi_var'] / (val['sb_wifi_var']+0.0001)
val['category_id'] = val['category_id'].map(lambda x:str(x).split('_')[1])
val['mall_id'] = val['mall_id'].map(lambda x:str(x).split('_')[1])

# val['sw_average_ratio'] = (val['c_sw_average'] + 0.5 ) / (val['sw_average'] + 1)
# val['bw_average_ratio'] = (val['c_bw_average'] + 0.5 ) / (val['bw_average'] + 1)

# val['price'] = np.log1p(val['price'])

print(val.head())

train_val_label = val.pop('label')
y_test = train_val_label.values
X_test = val[features].values
del train_val_label;gc.collect()

import lightgbm as lgb
# load or create your dataset
# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
del X_train;gc.collect()
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

#################################################################################################################
print('Start training...')

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc'},
    'num_leaves': 256,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
}

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=2000,
                valid_sets=lgb_eval,
                early_stopping_rounds=15)

result_test = gbm.predict(X_test, num_iteration=gbm.best_iteration)

result1 = pd.DataFrame({'pre_p':list(result_test)})
result1 = pd.concat([val[['shop_id','index','shop_id_y']],result1],axis=1)
result1 = pd.DataFrame(result1).sort_values('pre_p',ascending=False).drop_duplicates(['index'])
print(sum(result1['shop_id'] == result1['shop_id_y']) * 1.0 / len(result1['shop_id']))
# del train;gc.collect()
del val;gc.collect()
del result1;gc.collect()

# 保存model
gbm.save_model('model.txt')

####################################################测试数据进行分类##############################################
dir = '../input/'
shop_info_tmp = pd.read_csv(dir + 'ccf_first_round_shop_info.csv')
shop_info = shop_info_tmp[['shop_id','category_id','longitude','latitude','price']]
del shop_info_tmp;gc.collect()
shop_info.rename(columns={'longitude':'s_longitude','latitude':'s_latitude'},inplace=True)

print('Load model to predict')

gbm = lgb.Booster(model_file='model.txt')

print('reading train/val data ')
sub = pd.read_csv(dir + 'sub_wifi.csv')
print(sub.shape)
sub_user_info = pd.read_csv(dir + 'evaluation_public.csv')
# row_id,user_id,mall_id,time_stamp,longitude,latitude,wifi_infos
sub_user_info = sub_user_info[['row_id','longitude','latitude','time_stamp']]
sub = pd.merge(sub,shop_info,on=['shop_id'],how='left')
del sub['label']
del shop_info;gc.collect()

sub = sub.dropna()
sub = pd.merge(sub,sub_user_info,on=['row_id'],how='left')
print(sub.head)
sub['time_stamp'] = pd.to_datetime(sub['time_stamp'])
sub['current_hour'] =  pd.DatetimeIndex(sub.time_stamp).hour
# sub['current_week'] =  pd.DatetimeIndex(sub.time_stamp).dayofweek

sub['distance'] = hafuman_km(sub['s_longitude'],sub['s_latitude'],sub['longitude'],sub['latitude'])
sub['current_bearing_array'] = bearing_array(sub.s_latitude.values, sub.s_longitude.values,
                                                                                    sub.latitude.values, sub.longitude.values)

#sub['distance'] = np.log1p(sub['distance'])
sub['c_wifi_var'] = sub['strength'] - sub['c_sw_average']
sub['wifi_var'] = sub['strength'] - sub['s_avg_power']
sub['s_wifi_var'] = sub['strength'] - sub['w_avg_power']
sub['sb_wifi_var'] = sub['strength'] - sub['sb_history_avg_power']
sub['c_sb_wifi_var'] = sub['strength'] - sub['c_sb_history_avg_power']

sub['angle_var'] = sub['history_bearing_array_median'] - sub['current_bearing_array']

sub['s_sb_wifi_var_ratio'] = sub['c_sb_wifi_var'] / (sub['sb_wifi_var']+0.0001)

sub['category_id'] = sub['category_id'].map(lambda x:str(x).split('_')[1])
sub['mall_id'] = sub['mall_id'].map(lambda x:str(x).split('_')[1])
print(sub.head())
sub_r_s = sub[['row_id','shop_id']]
sub_ = sub[features]
del sub;gc.collect()
print(sub_.columns)

sub_lgb = sub_[features].values

result = gbm.predict(sub_lgb)
result = pd.DataFrame({'pre_p':list(result)})
result = pd.concat([sub_r_s,result],axis=1)
del sub_;gc.collect()
del sub_r_s;gc.collect()
result = pd.DataFrame(result).sort_values('pre_p',ascending=False).drop_duplicates('row_id')
print(result.shape)
result[['row_id','shop_id','pre_p']].to_csv('./tmp.csv',index=None)

result = pd.merge(result[['row_id','shop_id','pre_p']],sub_user_info,on=['row_id'],how='left')
result = result[['row_id','shop_id']]
result = result.fillna('s_167275')
result[['row_id','shop_id']].to_csv('./result.csv',index=None)