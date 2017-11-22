# *coding=utf-8*
import xgboost as xgb
import pickle
import pandas as pd
import os
import math
import warnings
from tqdm import tqdm
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
warnings.filterwarnings('ignore')
SHOP_DATA_PATH = '/home/moric/loc/data/ccf_first_round_shop_info.csv'
USER_DATA_PATH = '/home/moric/loc/data/ccf_first_round_user_shop_behavior.csv'
TEST_DATA_PATH = '/home/moric/loc/data/evaluation_public.csv'
shop = pd.read_csv(SHOP_DATA_PATH)
user = pd.read_csv(USER_DATA_PATH)
test = pd.read_csv(TEST_DATA_PATH)

df_shop = shop.copy()
df_shop.columns = ['shop_id', 'category_id', 'shop_longitude', 'shop_latitude', 'price', 'mall_id']
df_user = pd.merge(user, df_shop, how='left', on='shop_id').loc[:, ['shop_id', 'mall_id', 'longitude', 'latitude', 'wifi_infos']]

mall_bssid = pd.merge(user, df_shop, on='shop_id', how='left').loc[:, ['mall_id', 'wifi_infos']]
mall_bssid_map = pickle.load(open('/home/moric/loc/cache/mall_bssid_map', 'rb'))
mall_bssid_list = list(set(shop['mall_id'].tolist()))


test_list = list(set(test['mall_id'].tolist()))
for mall_id in tqdm(test_list[:1]):
    mall_bssid_map_util = mall_bssid_map[mall_id].copy()
    train_mall = df_user.loc[df_user.mall_id == mall_id]
    for wifi_infos in train_mall['wifi_infos'].tolist():
        wifi_info = wifi_infos.split(';')
        for wifi in wifi_info:
            bssid = wifi.split('|')[0]
            mall_bssid_map_util[bssid] += 1
    X_bssid_col = list(dict(sorted(mall_bssid_map_util.items(), key=lambda d:d[1], reverse=True)).keys())
    X_bssid_dict = {}
    for bssid in X_bssid_col: X_bssid_dict[bssid] = 0
    X_train_bssid_sig = []
    for wifi_infos in train_mall['wifi_infos'].tolist():
        temp = X_bssid_dict.copy()
        wifi_info = wifi_infos.split(';')
        for wifi in wifi_info:
            if wifi.split('|')[0] in X_bssid_col:
                if wifi.split('|')[2] == 'true':
                    temp[wifi.split('|')[0]] = 1
                else:
                    temp[wifi.split('|')[0]] = math.pow(1.01, (int(wifi.split('|')[1])))
        X_train_bssid_sig.append(list(temp.values()))
    X_gps = train_mall.loc[:, ['longitude', 'latitude']].reset_index(drop=True)
    chi = SelectKBest(chi2, k=1000).fit(X_train_bssid_sig, Y_train)
    X_wifi = chi.transform(X_train_bssid_sig)
    X_train = pd.concat([pd.DataFrame(X_wifi), X_gps], axis=1)
    Y_train = train_mall['shop_id'].reset_index(drop=True)
    le = preprocessing.LabelEncoder()
    le.fit(Y_train.tolist())
    Y_train = le.transform(Y_train.tolist())

    X_test_bssid_sig = []
    test_mall = test.loc[test.mall_id == mall_id].loc[:, ['row_id', 'mall_id', 'longitude', 'latitude', 'wifi_infos']].reset_index(drop=True)
    for wifi_infos in test_mall['wifi_infos'].tolist():
        temp = X_bssid_dict.copy()
        wifi_info = wifi_infos.split(';')
        for wifi in wifi_info:
            if wifi.split('|')[0] in X_bssid_col:
                if wifi.split('|')[2] == 'true':
                    temp[wifi.split('|')[0]] = 1
                else:
                    temp[wifi.split('|')[0]] = math.pow(1.01, (int(wifi.split('|')[1])))
        X_test_bssid_sig.append(list(temp.values()))
    X_test = chi.transform(X_test_bssid_sig)
    X_test = pd.concat([pd.DataFrame(X_test), test_mall.loc[:, ['longitude', 'latitude']].reset_index(drop=True)], axis=1)

    num_class = Y_train.max() + 1
    params = {
        'objective': 'multi:softmax',
        'eta': 0.1,
        'max_depth': 9,
        'eval_metric': 'merror',
        'seed': 0,
        'num_class': num_class,
        'silent': 1
    }
    xgb_train = xgb.DMatrix(X_train, label=Y_train)
    xgb_test = xgb.DMatrix(X_test)
    watchlist = [(xgb_train, 'train'), (xgb_test, 'test')]
    num_rounds = 60
    clf = xgb.train(params, xgb_train, num_rounds, watchlist, early_stopping_rounds=10)

    pre = clf.predict(xgb_test)
    test_mall['pre'] = pre
    test_mall['pre_shop'] = test_mall['pre'].apply(lambda x: le.inverse_transform(int(x)))
    sub_mall = pd.DataFrame()
    sub_mall['row_id'] = test_mall['row_id']
    sub_mall['shop_id'] = test_mall['pre_shop']
    sub_mall.to_csv('/home/moric/loc/ff/' + mall_id + '.csv', index=False)


import os
file = os.listdir('/home/moric/loc/ff/')
sub = pd.DataFrame()
for f in file:
    ret = pd.read_csv('/home/moric/loc/ff/' + f)
    sub = pd.concat([sub, ret], axis=0)
sub['predict_shop'] = sub['shop_id']
pre = pd.read_csv('/home/moric/loc/sub/0.1865.csv')
sub = pd.merge(pre, sub, how='left', on='row_id')
sub = sub.loc[:, ['row_id', 'predict_shop']]
sub.columns = ['row_id', 'shop_id']
sub.to_csv('/home/moric/loc/sub/new_xgb_60.csv', index=False)
