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

def bearing_array(lat1, lng1, lat2, lng2):
    """ function was taken from beluga's notebook as this function works on array
    while my function used to work on individual elements and was noticably slow"""
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))


def get_features_list():
    features = [
                # 原始特征 0 - 4
                'bsCount','category_id','mall_id','price','connect',
                # 偏差特征 5 - 10 
                'c_wifi_var','wifi_var','angle_var','sb_wifi_var','s_wifi_var','c_sb_wifi_var',
                # 'sb_history_avg_power',c_sb_history_avg_power'
                # 时间特征 11
                'current_hour',
                # 范围角度特征 12 - 15
                'distance','s_median_scale','current_bearing_array','history_bearing_array_median',
                # 统计特征 16
                'wifi_cover_shop',
                # 比例特征 17 - 18
                'shop_wifi_connect_ratio','tfid_features',
                # 其他特征 19 - 20
                'nature_order',
                'w_std_power',
                # 
                's_sb_wifi_var_ratio',
                # 时间历史热度
                # '1_shop_hot','3_shop_hot', '5_shop_hot', '7_shop_hot'

                #我们自己添加的feature
                'p_avg_price',
                'p_shop_favor',
                's_hot',
                'r_mean',
                'wifirank_diff',
                's_conncectwifi_count'

                ]
      
    return features

def time_map(time):
    time = int(time)
    if (time >= 23) | (time <=6):
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
