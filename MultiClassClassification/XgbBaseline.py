# *coding=utf-8*
# 根据商场mall进行划分，减少特征编码
import pandas as pd
import numpy as np
from sklearn import  preprocessing
import xgboost as xgb
import gc

path='../input/'
df=pd.read_csv(path+u'ccf_first_round_user_shop_behavior.csv')
shop=pd.read_csv(path+u'ccf_first_round_shop_info.csv')
test=pd.read_csv(path+u'evaluation_public.csv')
df=pd.merge(df,shop[['shop_id','mall_id']],how='left',on='shop_id')
df['time_stamp']=pd.to_datetime(df['time_stamp'])
train=pd.concat([df,test])

mall_list=list(set(list(shop.mall_id)))
result=pd.DataFrame()
for mall in mall_list:

    #对m_7800和m_7168进行单独调参
    if(mall=='m_7800') | (mall=='m_7168'):
        learning_ratio =0.01
        num_rounds = 350
        #depth = 10
    else:
        learning_ratio =0.1
        num_rounds = 150
        #depth = 9
    train1=train[train.mall_id==mall].reset_index(drop=True)       
    l=[]
    wifi_dict = {}

    ### 剔除掉无效的移动wifi，设置频次阈值
    for index,row in train1.iterrows():
        r = {}
        wifi_list = [wifi.split('|') for wifi in row['wifi_infos'].split(';')]
        for i in wifi_list:
            r[i[0]]=int(i[1])
            if i[0] not in wifi_dict:
                wifi_dict[i[0]]=1
            else:
                wifi_dict[i[0]]+=1
        l.append(r)    
    delate_wifi=[]
    for i in wifi_dict:
        if wifi_dict[i]<20:
            delate_wifi.append(i)
    m=[]
    for row in l:
        new={}
        for n in row.keys():
            if n not in delate_wifi:
                new[n]=row[n]
        m.append(new)
    train1 = pd.concat([train1,pd.DataFrame(m)], axis=1)
    df_train=train1[train1.shop_id.notnull()]
    df_test=train1[train1.shop_id.isnull()]
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(df_train['shop_id'].values))
    df_train['label'] = lbl.transform(list(df_train['shop_id'].values))    
    num_class=df_train['label'].max()+1    
    params = {
            'objective': 'multi:softmax',
            'eta': learning_ratio,
            'max_depth': 9,
            'eval_metric': 'merror',
            'seed': 0,
            'missing': -999,
            'num_class':num_class,
            'silent' : 1
            }
    print(train1.columns)
    feature=[x for x in train1.columns if x not in ['user_id','label','shop_id','time_stamp','mall_id','wifi_infos']]
    print(feature)
    print(df_test[feature])
    xgbtrain = xgb.DMatrix(df_train[feature], df_train['label'])
    xgbtest = xgb.DMatrix(df_test[feature])
    watchlist = [ (xgbtrain,'train'), (xgbtrain, 'test') ]

    model = xgb.train(params, xgbtrain, num_rounds, watchlist, early_stopping_rounds=15)
    df_test['label']=model.predict(xgbtest)
    df_test['shop_id']=df_test['label'].apply(lambda x:lbl.inverse_transform(int(x)))
    r=df_test[['row_id','shop_id']]
    result=pd.concat([result,r])
    result['row_id']=result['row_id'].astype('int')
    result.to_csv(path+'result.csv',index=False)