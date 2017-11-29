# *coding=utf-8*
# |8月4日-17日为训练集的候选集|8月18日-8月24日构造线下训练集|
# |8月11日-24日为验证集的候选集|8月25日-8月31日构造线下验证集|
# |8月18日-31日为测试集的候选集|9月为测试数据集|
# 使用规则将候选集和训练集，验证集，测试集进行匹配


from tools import hafuman_km,mode_function,bearing_array
import math

import numpy as np
import pandas as pd
import gc



# 使用规则将候选集和训练集，验证集，测试集进行匹配
def make_features(feature,org_data,mode= list([4,4,6]),index='index',row_id='index'):

    #距离规则
    #train_wifi_train_candidate_F = feature.sort_values([index, 'history_bearing_array_median'], ascending=False).groupby([index],as_index=False).head(5)

    #第一个规则，选出候选集每条消费记录中WiFi强度的top4
    train_wifi_train_candidate = feature.sort_values([index, 'strength'], ascending=False).groupby([index], as_index=False
                                                                                                   ).head(mode[0])
    #第二个规则，选择候选集中出现bssid - shopid 的频次top4
    train_wifi_train_candidate_tmp = train_wifi_train_candidate.groupby(['bssid', 'shop_id'], as_index=False).strength.agg(
        {'bsCount': 'count'})
    #将频次加入候选特征
    train_wifi_train_candidate = pd.merge(train_wifi_train_candidate_tmp,train_wifi_train_candidate,on=['bssid','shop_id'],how='left')
    train_wifi_train_candidate = train_wifi_train_candidate.sort_values(['bssid', 'bsCount'], ascending=False)
    #选出每个WiFi出现的频次
    train_wifi_train_candidate = train_wifi_train_candidate.groupby(['bssid'], as_index=False).head(mode[1])
    train_wifi_train_candidate.drop(['connect','index','mall_id','strength','time_stamp'],axis=1,inplace=True)

    #选择待匹配数据供候选的bassid，WiFi强度的top6
    train_wifi_test_candidate = org_data.sort_values([row_id, 'strength'], ascending=False).groupby([row_id],as_index=False
                                                                                                    ).head(mode[2])
    most_prop = pd.merge(train_wifi_train_candidate, train_wifi_test_candidate, how='right', on='bssid')

    # 制作标签，候选集的4个bssid - shopid与训练集的6个bssid - shopid进行匹配，shop_id相同的label=1
    most_prop['connect'] = most_prop['connect'].astype(int)
    if row_id == 'row_id':  #测试数据与候选集匹配后没有label
        most_prop['label'] = np.nan
    else:
        most_prop['label'] = most_prop.shop_id_y == most_prop.shop_id_x
        most_prop['label'] = most_prop['label'].astype(int)
    return most_prop

# 制作候选特征
def get_one_feature(count_feature):
    dir = '../data/'
    user_behavier = pd.read_csv(dir + 'ccf_first_round_user_shop_behavior.csv')[['longitude','latitude']].reset_index()
    shop_info_tmp = pd.read_csv(dir + 'ccf_first_round_shop_info.csv')[['shop_id','price','longitude','latitude','category_id']]
    # 加入类别
    shop_info_tmp['category_id'] = shop_info_tmp['category_id'].map(lambda x:str(x).split('_')[1])

    shop_info_tmp.rename(columns={'latitude':'s_latitude','longitude':'s_longitude'},inplace=True)

    count_feature_with_shop_price = pd.merge(count_feature,shop_info_tmp,on=['shop_id'],how='left')
    #候选特征count_feature中是经过处理的，加入经纬度
    count_feature_with_shop_price_position = pd.merge(count_feature_with_shop_price,user_behavier,on=['index'],how='left')



    #|||||||||||||||||||||||||||||||||||||||||| feature engineering |||||||||||||||||||||||||||||||||||||||||||||

    user_shop_behavior = pd.read_csv(dir + 'ccf_first_round_user_shop_behavior.csv').reset_index()
    user_merge_shop = pd.merge(user_shop_behavior,shop_info_tmp,on=['shop_id'],how='left')


    ############################### 用户特征 #########################################
    #1.人均消费水平
    people_avg_price = user_merge_shop.groupby(['user_id'],as_index=False).price.agg({'p_avg_price':np.mean})
    count_feature = pd.merge(count_feature,people_avg_price,on=['user_id'],how='left')
    del people_avg_price;gc.collect()


    ############################### 用户和shop的组合特征 ###############################
    #1.人去店的购物喜好
    people_shop_favor = user_merge_shop.groupby(['user_id','shop_id'],as_index=False).shop_id.agg(
            {'p_shop_favor':'count'})
    #people_shop_favor.rename(columns={'shop_id':'people_shop_favor'},inplace=True)
    count_feature = pd.merge(count_feature,people_shop_favor,on=['user_id','shop_id'],how='left')

    del people_shop_favor;gc.collect()

    #2.人去店的中位数位置
    people_shop_location_longitude = count_feature.groupby(['user_id','shop_id'],as_index=False).longitude.agg(
            {'p_shop_location_longitude':np.median})
    people_shop_location_latitude = count_feature.groupby(['user_id','shop_id'],as_index=False).latitude.agg(
            {'p_shop_location_latitude':np.median})
    count_feature = pd.merge(count_feature,people_shop_location_longitude,on=['user_id','shop_id'],how='left')
    count_feature = pd.merge(count_feature,people_shop_location_latitude,on=['user_id','shop_id'],how='left')
    del people_shop_location_longitude;gc.collect()
    del people_shop_location_latitude;gc.collect()
    count_feature.drop(['longitude','latitude','s_longitude','s_latitude'],axis=1)

    #3.shop的购物记录数，即shop的人气程度
    shop_hot = user_merge_shop.groupby(['shop_id'],as_index=False).user_id.agg(
            {'s_hot':'count'}
    )
    count_feature = pd.merge(count_feature,shop_hot,on=['shop_id'],how='left')

    #4.用户与店铺距离特征
    count_feature_with_shop_price_position['distance_history'] = hafuman_km(count_feature_with_shop_price_position['s_longitude'],
                                                                            count_feature_with_shop_price_position['s_latitude'],
    																		count_feature_with_shop_price_position['longitude'],
                                                                            count_feature_with_shop_price_position['latitude'])

    count_feature_with_shop_price_position['history_bearing_array'] = bearing_array(count_feature_with_shop_price_position.s_latitude.values,
                                                                                    count_feature_with_shop_price_position.s_longitude.values,
    																				count_feature_with_shop_price_position.latitude.values,
                                                                                    count_feature_with_shop_price_position.longitude.values)

    count_feature = pd.merge(count_feature,count_feature_with_shop_price_position[['shop']])


    del user_behavier;gc.collect()
    del count_feature_with_shop_price;gc.collect()

    ######################################### shop特征 #########################################
    print(count_feature_with_shop_price_position.columns)
    wifirank_mean = count_feature_with_shop_price_position.groupby(['shop_id','bssid'],as_index=False).nature_order.agg(
            {'r_mean':np.mean})
    count_feature = pd.merge(count_feature,wifirank_mean,on=['shop_id','bssid'],how='left')
    count_feature['wifirank_diff'] = count_feature['nature_order'] - count_feature['r_mean']
    #count_feature = count_feature.drop(['r_mean'],axis=1)
    del wifirank_mean;gc.collect()

    print(count_feature.columns)
    #每个店被连接的WiFi数量
    shop_conncectwifi_count = count_feature[count_feature['connect'] == 1].groupby(['shop_id'],as_index=False).bssid.agg(
            {'s_conncectwifi_count':'count'})
    count_feature = pd.merge(count_feature,shop_conncectwifi_count,on=['shop_id'],how='left')
    del shop_conncectwifi_count;gc.collect()


    # 1 shop 的行为范围中位数（测试发现中位数效果好）
    shop_scale = count_feature_with_shop_price_position.groupby(['mall_id','shop_id'],as_index=False).distance_history.agg(
            {'s_median_scale':np.median})
    count_feature = pd.merge(count_feature,shop_scale,on=['mall_id','shop_id'],how='left')
    del shop_scale;gc.collect()
    # 1.2 shop 的行为角度特征，表示历史的方向特征 // 加了当前和历史的差值特征--特征效果较好 同时保留当前特征
    shop_degree = count_feature_with_shop_price_position.groupby(['mall_id','shop_id'],as_index=False).history_bearing_array.agg(
            {'history_bearing_array_median':np.median})
    count_feature = pd.merge(count_feature,shop_degree,on=['mall_id','shop_id'],how='left')
    del shop_degree;gc.collect()
   
    # 强度特征 均值
    # 2.1.历史商店周围的平均wifi强度 （目的：当前wifi强度 - 历史wifi强度均值）// 
    shop_around_wifi_power = count_feature_with_shop_price_position.groupby(['mall_id','shop_id'],as_index=False).strength.agg(
            {'s_avg_power':np.mean})
    count_feature = pd.merge(count_feature,shop_around_wifi_power,on=['mall_id','shop_id'],how='left')
    del shop_around_wifi_power;gc.collect()

    # 2.1.1 发生链接时的商铺周围的wifi平均强度
    number_count_shop_wifi_strength_c = count_feature[count_feature['connect'] == 1].groupby(['mall_id','shop_id'],as_index=False
                                                                                             ).strength.agg({'c_sw_average':np.mean})
    count_feature = pd.merge(count_feature,number_count_shop_wifi_strength_c,on=['mall_id','shop_id'],how='left')
    del number_count_shop_wifi_strength_c;gc.collect()

    # 2.2.历史商店和wifi组合时，周围的wifi强度
    shop_bssid_around_wifi_power = count_feature_with_shop_price_position.groupby(['mall_id','bssid','shop_id'],as_index=False
                                                                                  ).strength.agg({'sb_history_avg_power':np.mean})
    count_feature = pd.merge(count_feature,shop_bssid_around_wifi_power,on=['mall_id','shop_id','bssid'],how='left')
    del shop_bssid_around_wifi_power;gc.collect()
    
    
    # 2.3.历史商店和wifi组合时且连接时的wifi，周围的wifi强度
    shop_bssid_around_wifi_power_c = count_feature_with_shop_price_position[count_feature_with_shop_price_position['connect'] == 1
    ].groupby(['mall_id','bssid','shop_id'],as_index=False).strength.agg({'c_sb_history_avg_power':np.mean})
    count_feature = pd.merge(count_feature,shop_bssid_around_wifi_power_c,on=['mall_id','shop_id','bssid'],how='left')
    del shop_bssid_around_wifi_power_c;gc.collect()
    
    del count_feature_with_shop_price_position;gc.collect()

    # 2.4 商场中 wifi 的强度特征
    wifi_power_feat = count_feature.groupby(['mall_id','bssid'],as_index=False).strength.agg(
            {'w_avg_power':np.mean,'w_std_power':np.std})
    count_feature = pd.merge(count_feature,wifi_power_feat,on=['mall_id','bssid'],how='left')
    del wifi_power_feat;gc.collect()



    # 3.1 wifi被链接的次数
    wifi_is_connected_times = count_feature[count_feature['connect'] == 1].groupby(['mall_id','bssid'],as_index=False
                                                                                   ).strength.count()
    wifi_is_connected_times.rename(columns={'strength':'wifi_is_connected_times'},inplace=True)
    count_feature = pd.merge(count_feature,wifi_is_connected_times,on=['mall_id','bssid'],how='left')
    del wifi_is_connected_times;gc.collect()

    # 3.2 wifi被链接时与商铺发生的次数
    wifi_is_connected_shop_times = count_feature[count_feature['connect'] == 1].groupby(['mall_id','bssid','shop_id'],as_index=False
                                                                                        ).strength.count()
    wifi_is_connected_shop_times.rename(columns={'strength':'wifi_is_connected_shop_times'},inplace=True)
    count_feature = pd.merge(count_feature,wifi_is_connected_shop_times,on=['mall_id','shop_id','bssid'],how='left')
    del wifi_is_connected_shop_times;gc.collect()

    count_feature['shop_wifi_connect_ratio'] = count_feature['wifi_is_connected_shop_times'] / (count_feature['wifi_is_connected_times'] + 1.0 )

    # 3.3 wifi覆盖的shop个数
    wifi_cover_count = count_feature.groupby(['mall_id','bssid'],as_index=False).shop_id.apply(lambda x : len(set(x))).reset_index()
    wifi_cover_count.rename(columns={0:'wifi_cover_shop'},inplace=True)
    count_feature = pd.merge(count_feature,wifi_cover_count,on=['mall_id','bssid'],how='left')
    del wifi_cover_count;gc.collect()

    # tfidf-特征 wifi 的tfidf统计特征

    # 3.4 wifi和shop出现的次数
    wifi_shop_count = count_feature.groupby(['mall_id','shop_id','bssid'],as_index=False).strength.count()
    wifi_shop_count.rename(columns={'strength':'wifi_shop_count'},inplace=True)
    count_feature = pd.merge(count_feature,wifi_shop_count,on=['mall_id','shop_id','bssid'],how='left')
    del wifi_shop_count;gc.collect()

    # 3.5 shop有关的wifi个数
    wifi_shop_length = count_feature.groupby(['mall_id','shop_id'],as_index=False).bssid.count()
    wifi_shop_length.rename(columns={'bssid':'wifi_shop_length'},inplace=True)
    count_feature = pd.merge(count_feature,wifi_shop_length,on=['mall_id','shop_id'],how='left')
    del wifi_shop_length;gc.collect()

    count_feature['wifi_shop_ratio_tfidf'] = count_feature['wifi_shop_count'] / (count_feature['wifi_shop_length'] + 1.0)

    # 3.6 bssid个数
    mall_wifi_count = count_feature.groupby(['mall_id','bssid'],as_index=False).strength.count()
    mall_wifi_count.rename(columns={'strength':'mall_wifi_count'},inplace=True)
    count_feature = pd.merge(count_feature,mall_wifi_count,on=['mall_id','bssid'],how='left')
    del mall_wifi_count;gc.collect()

    # 3.7 商铺周围bssid的个数
    shop_around_count = count_feature.groupby(['mall_id','shop_id'],as_index=False).bssid.apply(lambda x : len(set(x))).reset_index()
    shop_around_count.rename(columns={0:'shop_around_count'},inplace=True)
    count_feature = pd.merge(count_feature,shop_around_count,on=['mall_id','shop_id'],how='left')
    del shop_around_count;gc.collect()



    count_feature['shop_around_ration_tfidf'] = count_feature['shop_around_count'] / (count_feature['mall_wifi_count'] + 1)

    count_feature['tfid_features'] = np.log1p(count_feature['shop_around_ration_tfidf']) * count_feature['wifi_shop_ratio_tfidf']

    count_feature['sun_features'] = count_feature['shop_around_count'] + count_feature['mall_wifi_count'] + count_feature['wifi_shop_count'] + count_feature['wifi_shop_length']
    # 构造集合特征

    # 时间串信息组合
    count_feature['time_stamp'] = pd.to_datetime(count_feature['time_stamp'])
    count_feature['history_hour'] =  pd.DatetimeIndex(count_feature.time_stamp).hour
    count_feature['history_day'] =  pd.DatetimeIndex(count_feature.time_stamp).day
    
    count_feature = count_feature.fillna(0)
    count_feature.rename(columns={'nature_order':'history_nature_order'},inplace=True)
    return count_feature


#读取数据
dir = '../input/'
user_shop_behavior = pd.read_csv(dir + 'ccf_first_round_user_shop_behavior.csv').reset_index()
train_wifi = pd.read_csv( dir + 'train_wifi.csv')
#处理过的训练数据添加原始数据的时间戳，加入时间信息
train_wifi = pd.merge(train_wifi,user_shop_behavior[['index','time_stamp']])
del user_shop_behavior;gc.collect()


#划分线下训练集
offline_train = train_wifi[(train_wifi['time_stamp'] < '2017-08-25 00:00')&(train_wifi['time_stamp'] >= '2017-08-18 00:00' )]
#划分训练集的候选集
offline_train_feature = train_wifi[(train_wifi['time_stamp'] < '2017-08-18 00:00')&(train_wifi['time_stamp'] >= '2017-08-04 00:00' )]
#构造候选集的候选特征
offline_train_feature = get_one_feature(offline_train_feature)
#将候选集与训练集进行匹配
offline_train = make_features(offline_train_feature,offline_train)

print(offline_train.head())
print(offline_train.columns)
offline_train.to_csv(dir + 'offline_train.csv',index=False)
del offline_train;gc.collect()


#划分线下验证集
offline_val = train_wifi[(train_wifi['time_stamp'] < '2017-09-01 00:00')&(train_wifi['time_stamp'] >= '2017-08-25 00:00' )]
#划分验证集的候选集
offline_val_feature = train_wifi[(train_wifi['time_stamp'] < '2017-08-25 00:00')&(train_wifi['time_stamp'] >= '2017-08-11 00:00' )]
#构造候选集的候选特征
offline_val_feature = get_one_feature(offline_val_feature)
#将候选集与验证集进行匹配
offline_val = make_features(offline_val_feature,offline_val)

print(offline_val.head())
print(offline_val.columns)
offline_val.to_csv(dir + 'offline_val.csv',index=False)
del offline_val;gc.collect()


sub_wifi = pd.read_csv(dir + 'test_wifi.csv')
#划分线上测试集的候选集
online_sub_feature = train_wifi[(train_wifi['time_stamp'] < '2017-09-01 00:00')&(train_wifi['time_stamp'] >= '2017-08-18 00:00' )]
#构造候选集的候选特征
online_sub_feature = get_one_feature(online_sub_feature)
#将候选集与测试集进行匹配
sub_wifi = make_features(online_sub_feature,sub_wifi,row_id='row_id')

print(sub_wifi.head())
print(sub_wifi.columns)
sub_wifi.to_csv(dir + 'sub_wifi.csv',index=False)
del sub_wifi;gc.collect()
