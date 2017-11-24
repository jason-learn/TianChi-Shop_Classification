#coding:utf-8
import numpy as np
from scipy.stats import mode 
# 根据经纬度计算距离
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km
# 曼哈顿距离
def hafuman_km(lon1, lat1, lon2, lat2):
    return haversine_np(lon1,lat1,lon2,lat1) + haversine_np(lon2,lat1,lon2,lat2)

def mode_function(df):
    # print df
    df = (df*10000)
    df = df.astype(int)
    # print df
    counts = mode(df) 
    return counts[0][0] / 10000


def get_features_list():

	features = [
                # 原始特征 0 - 4
                'bsCount','category_id','mall_id','price','connect',
                # 时间特征 11
                'current_hour',
                # 范围 12 - 15
                'distance','s_median_scale',
                # 统计特征 16
                'wifi_cover_shop',
                # 比例特征 17 - 18
                'shop_wifi_connect_ratio','tfid_features',

                ]
      
	return features

def time_map(time):
    time = int(time)
    if (time >= 23) & (time <=6):
        return 0
    elif (time >=7) & (time<=10):
        return 1
    elif (time >=11) & (time <=14):
        return 2
    elif (time >=15) & (time <= 18):
        return 3
    elif (time >=19) & (time <=21):
        return 4
    else:
        return 5
