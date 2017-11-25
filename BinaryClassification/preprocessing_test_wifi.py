# *coding=utf-8*

import pandas as pd
from tqdm import tqdm

'''
读取数据
'''
print('get_test_wifi')
dir = '../input/'
test_behavior = pd.read_csv(dir + 'evaluation_public.csv')

# 提取wifi信息，分割为 bassid wifi_strength 格式
def get_wifi_dict(df):
    '''
    :param df: 待处理数据
    :return: 将每条记录的所有WIFI分开成多个行并返回
    '''
    # 构造WiFi字典
    wifiDict={
        # 训练集需要添加shop_id，测试集不需要
        'user_id':[],
        'row_id': [],
        'bssid': [],
        'strength': [],
        'connect': [],
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
        # print(index)
        order_index = 1
        for wifi in row.wifi_infos.split(';'):
            info = wifi.split('|')
            if(info[0] in delate_wifi):
                continue

            #当WiFi是店铺固定WiFi，处理数据
            else:
                wifiDict['user_id'].append(row.user_id)
                wifiDict['row_id'].append(row.row_id)
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

#因为要分类的就是shop_id,所以测试集无法进行表的连接，根据每条记录的WiFi信息进行分开处理
test_wifi = get_wifi_dict(test_behavior)
test_wifi.to_csv(dir + 'test_wifi.csv', index=False)