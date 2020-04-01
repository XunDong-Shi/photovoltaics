#!/usr/bin/env python
# coding: utf-8

# # 0.从外部传入参数

# In[ ]:


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('Number',type=int,help='输入电站序号')
parser.add_argument('Cp',type=int,help='输入电站额定功率')
args = parser.parse_args()
num=args.Number
cp=args.Cp
print('正在处理数据集{}，装机功率为{}'.format(num,cp))


# In[ ]:


Pca=True
subsect=False


# # 1.1文件读取

# In[2]:


import os
import sys
import warnings
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"


# In[3]:


import pandas as pd
file_path="J:/人工智能学习/数据集/【更新】国能日新竞赛数据/国能日新竞赛数据/"


# In[4]:


import pandas as pd
data=pd.read_csv(file_path+'final_data/final_data'+str(num)+'.csv',encoding='gbk',parse_dates=['时间','ds'])


# ## 1.2 是否采用分段预测的方法

# In[5]:


if subsect:
    data=data[data['辐照度']!=0].reset_index(drop=True)
else:
    pass


# In[6]:


spilt_num=data[data['id'].isnull()].index[-1]+1


# ## 2.1 增加连续值多项式特征

# ### 2.1.1先定义分类和连续值变量

# In[8]:


not_use=[ '时间', 'mday','year', 'id', '年月日', '当日时间', 'ds', '小时', '是否小于阈值','delta_time','时间_last']


# In[9]:


class_col=['总辐射是否为-1','辐照度小于0.9','类别风向','season','总辐射是否为-1_last','是否为断点']


# In[10]:


contin_col=['azimuth', 'altitude', 'diameter', 'distance', 'declination',
            'rightAscension', 'hours_float', 'yday', 'month', 
            '直辐射', '总辐射','yhat_lower', 'yhat', 'yhat_upper',
            '温度_month_mean', '温度_mean_delta', '湿度_month_mean', '湿度_mean_delta', 
            '辐照度_month_mean','辐照度_mean_delta', '风速_month_mean',
            '风速_mean_delta', '压强_month_mean','压强_mean_delta', 
            '风向_month_mean', '风向_mean_delta','温度_last','湿度_last', 
            '辐照度_last', '风速_last', '压强_last', '风向_last', 
             '温度_last_delta', '湿度_last_delta', '辐照度_last_delta',
            '风速_last_delta', '压强_last_delta', '风向_last_delta']


# ### 2.1.2 对部分连续值生成多项式特征

# In[11]:


from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2,include_bias=False,interaction_only=True)
poly_col=['压强','温度','湿度','辐照度','风向','风速']
polyed_array=poly.fit_transform(data[poly_col])


# ## 2.2 对连续值特征进行标准化

# In[12]:


import numpy as np
temp_data=np.concatenate((data[contin_col].values,polyed_array),axis=1)


# In[13]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
scaled_array=scaler.fit_transform(temp_data)


# ## 2.3 对类别特征进行编码

# In[14]:


from sklearn.preprocessing import OneHotEncoder
enc=OneHotEncoder()
encode_array=enc.fit_transform(data[class_col]).toarray()


# In[15]:


final_array=np.concatenate((scaled_array,polyed_array),axis=1)


# ## 2.4 对所有所有特征进行降维

# In[17]:


train_x=final_array[0:spilt_num]
test_x=final_array[spilt_num:]
train_y=data['实际功率'][0:spilt_num].values


# In[18]:


from sklearn.decomposition import PCA
if Pca==True:
    pca=PCA(n_components=0.95)
    pca.fit(final_array)
    train_x=pca.transform(train_x)
    test_x=pca.transform(test_x)
print('PCA处理后特征维度：',train_x.shape[1])


# # 3. 模型建立

# In[20]:


print('正在对数据集{}进行建模'.format(num))


# ## 3.1 XgBoost模型

# In[22]:


from sklearn.metrics import accuracy_score 
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


# In[23]:


import xgboost as xgb
from xgboost.sklearn import XGBRegressor
print('XGB模型')
model_xgb=XGBRegressor(eval_metric='mae',random_state=2020)
param_xgb={ 'max_depth': [4,6],'learning_rate': [0.05,0.1],'n_estimators': [200,500]}
gsearch_xgb=GridSearchCV(model_xgb,param_grid=param_xgb,scoring='neg_mean_absolute_error',cv=4)
gsearch_xgb.fit(train_x,train_y)
best_score_xgb=-gsearch_xgb.best_score_/cp
print("Best score: %0.3f" % best_score_xgb)
print("Best parameters set:")
best_parameters = gsearch_xgb.best_estimator_.get_params()
for param_name in sorted(param_xgb.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


# In[24]:


best_model_xgb=gsearch_xgb.best_estimator_
predict_y_xgb=best_model_xgb.predict(test_x).tolist()
print(60*'-')


# ## 3.3 Lasso模型

# In[25]:


from sklearn.linear_model import Lasso
model_lasso=Lasso(random_state=2020)
param_lasso={'alpha':[0.01,0.1,1]}
gsearch_lasso=GridSearchCV(model_lasso,param_grid=param_lasso,scoring='neg_mean_absolute_error',cv=4)
gsearch_lasso.fit(train_x,train_y)
best_score=-gsearch_lasso.best_score_/cp
print('Lasso模型')
print("Best score: %0.3f" % best_score)
print("Best parameters set:")
best_parameters = gsearch_lasso.best_estimator_.get_params()
for param_name in sorted(param_lasso.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


# In[26]:


best_model_lasso=gsearch_lasso.best_estimator_
predict_y_lasso=best_model_lasso.predict(test_x).tolist()
print(60*'-')


# ## 3.4 ElasticNet模型

# In[27]:


from sklearn.linear_model import ElasticNet
model_ela=ElasticNet(random_state=2020)
param_ela={'alpha':[0.05,0.2,1],
          'l1_ratio':[0.1,0.5,0.9]}
gsearch_ela=GridSearchCV(model_ela,param_grid=param_ela,scoring='neg_mean_absolute_error',cv=4)
gsearch_ela.fit(train_x,train_y)
best_score_ela=-gsearch_ela.best_score_/cp
print('Ela模型')
print("Best score: %0.3f" % best_score)
print("Best parameters set:")
best_parameters = gsearch_ela.best_estimator_.get_params()
for param_name in sorted(param_ela.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


# In[28]:


best_model_ela=gsearch_ela.best_estimator_
predict_y_ela=best_model_ela.predict(test_x).tolist()
print(60*'-')


# ## 3.5 SVR模型

# In[29]:


from sklearn import svm
model_svr=svm.SVR(kernel='rbf')
param_svr={'gamma':[0.05],
          'C':[1]}
gsearch_svr=GridSearchCV(model_svr,param_grid=param_svr,scoring='neg_mean_absolute_error',cv=4)
gsearch_svr.fit(train_x,train_y)
best_score_svr=-gsearch_svr.best_score_/cp
print('SVR模型')
print("Best score: %0.3f" % best_score_svr)
print("Best parameters set:")
best_parameters = gsearch_svr.best_estimator_.get_params()
for param_name in sorted(param_svr.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))


# In[30]:


best_model_svr=gsearch_svr.best_estimator_
predict_y_svr=best_model_svr.predict(test_x).tolist()
print(60*'-')


# ## 3.6 keras实现深度神经网络

# In[32]:


from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from keras.callbacks import EarlyStopping
def create_model(init='glorot_uniform',optimizer='adam'):
    model=Sequential()
    model.add(Dense(units=20,activation='relu',input_dim=train_x.shape[1],kernel_initializer=init))
    model.add(Dense(units=10,activation='relu',kernel_initializer=init))
    model.add(Dense(units=10,activation='relu',kernel_initializer=init))
    model.add(Dense(units=1,kernel_initializer=init))
    model.compile(loss='mean_absolute_error',optimizer='adam')
    return model
model_dnn=KerasRegressor(build_fn=create_model,epochs=200,batch_size=10,verbose=0)
x_train,x_valid,y_train,y_valid=train_test_split(train_x,train_y,
                                                 test_size=0.25,random_state=2020)
early_stopping=EarlyStopping(monitor='val_loss',patience=10,verbose=0)
history=model_dnn.fit(x_train,y_train,validation_data=(x_valid,y_valid),
                  callbacks=[early_stopping])


# In[33]:


predict_y_dnn=model_dnn.predict(test_x)
print(60*'-')


# ## 3.7 采用voteing的方法对不同模型预测值求平均数

# In[34]:


output_index=data['id'].dropna().values


# In[36]:


predict=pd.DataFrame({'id':output_index,'xgb':predict_y_xgb,
                        'lasso':predict_y_lasso,'svr':predict_y_svr,
                        'ela':predict_y_ela,'dnn':predict_y_dnn})


# In[37]:


predict['prediction']=predict[['xgb','lasso','ela','svr','dnn']].apply(lambda x:x.mean(),axis=1)


# In[40]:


limit=0.03*cp
if subsect:
    test_data=pd.read_csv(file_path+'test/test_{}.csv'.format(num),parse_dates=['时间'])
    temp=test_data[['id','辐照度']]
    predict=pd.merge(temp,predict,how='outer',left_on='id',right_on='id')
predict['prediction']=predict['prediction'].apply(lambda x: x if x>limit else limit)


# In[43]:


predict.to_csv(file_path+'all_predict/predict_'+str(num)+'.csv',encoding='gbk',index=False)


# # 7.结果输出

# In[42]:


import datetime as dt
import time
version=dt.datetime.now().strftime('%m%d%H%M')
predict[['id','prediction']].to_csv(file_path+'predict/'+'数据集_'+str(num)+'_'+version+'.csv',encoding='utf-8',index=False)


# In[ ]:


print('\n \n')

