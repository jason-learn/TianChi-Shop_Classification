# *coding=utf-8*

import pandas as pd
from tqdm import tqdm

'''
读取数据
'''
print('get_train_wifi')
dir = '../input/'
shop_info = pd.read_csv(dir + 'ccf_first_round_shop_info.csv')
train_behavior = pd.read_csv(dir + 'ccf_first_round_user_shop_behavior.csv')


# 提取wifi信息，分割为 bassid wifi_strength 格式
def get_wifi_dict(df):
    '''
    :param df: 原始数据
    :return: 将每条记录的所有WIFI分开成多个行并返回
    '''
    # 构造WiFi字典
    wifiDict={
        # 训练集需要添加shop_id，测试集不需要
        'user_id':[],
        'shop_id': [],
        'bssid': [],
        'strength': [],
        'connect': [],
        'index': [],
        'mall_id': [],
        'nature_order':[]
     }
    ### 剔除掉无效的移动wifi，设置频次阈值
    wifi_dict={}
    for index, row in tqdm(df.iterrows()):
        order_index = 1
        for wifi in row.wifi_infos.split(';'):
            info = wifi.split('|')
            if info[0] not in wifi_dict:
                wifi_dict[info[0]] = 1
            else:
                wifi_dict[info[0]] += 1
    delate_wifi =set([])
    for item in wifi_dict:
        if wifi_dict[item] < 15:
            delate_wifi.add(item)

    for index, row in tqdm(df.iterrows()):
        order_index = 1
        for wifi in row.wifi_infos.split(';'):
            info = wifi.split('|')
            if(info[0] in delate_wifi):
                continue

            #当WiFi是店铺固定WiFi，处理数据
            else:
                wifiDict['user_id'].append(row.user_id)
                wifiDict['shop_id'].append(row.shop_id)
                wifiDict['index'].append(index)
                wifiDict['mall_id'].append(row.mall_id)
                wifiDict['bssid'].append(info[0])
                wifiDict['strength'].append(info[1])
                wifiDict['connect'].append(info[2])
                wifiDict['nature_order'].append(order_index)
                order_index = order_index + 1  #WiFi在每条记录中的rank
    print('done')
    del df
    wifi = pd.DataFrame(wifiDict)
    return wifi

#将用户消费记录表和店铺信息表连接
train_behavior = pd.merge(train_behavior,shop_info,on=['shop_id'],how='left')
#根据每条记录的WiFi信息进行分开处理
train_wifi = get_wifi_dict(train_behavior)
train_wifi.to_csv(dir + 'train_wifi.csv', index=False)