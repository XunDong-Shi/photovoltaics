#!/usr/bin/env python
# coding: utf-8

# # 0.从外部传入参数
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('Number',type=int,help='输入电站序号')
parser.add_argument('Cp',type=int,help='输入电站额定功率')
args = parser.parse_args()
num=args.Number
cp=args.Cp
print('正在处理数据集{}，装机功率为{}'.format(num,cp))
# In[1]:



# # 1.文件读取

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


train_data=pd.read_csv(file_path+'data_new/train_new_{}.csv'.format(num),parse_dates=['时间'])
#train_data=train_data[train_data['辐照度']!=-1]
train_data['实际功率']=train_data['实际功率'].apply(lambda x:0 if x<0 else x)


# In[5]:


train_data.shape


# In[6]:


test_data=pd.read_csv(file_path+'data_new/test_new_{}.csv'.format(num),parse_dates=['时间'],encoding='gbk')
# test_data=test_data[test_data['辐照度']!=-1]
weather_data=pd.read_csv(file_path+'气象数据/电站{}_气象.csv'.format(num),parse_dates=['时间'],encoding='gbk')


# In[7]:


test_data.shape


# In[8]:


data=pd.concat([train_data,test_data],ignore_index=True)
weather_data.drop_duplicates(['时间'],inplace=True,keep='first')
data=data.merge(weather_data[['时间','直辐射','总辐射']],left_on=['时间'],right_on=['时间'])


# In[9]:


data.shape


# ## 2.1 根据官方要求清洗数据

# In[10]:


if num==7:
    train_data=train_data[(train_data["时间"] < "2018/03/01 00:00") |(train_data["时间"] > "2018/04/04 23:45")]
    train_data.reset_index(drop=True,inplace=True)
if num == 9:
    train_data=train_data[(train_data["时间"] < "2016/01/01 9:00") | (train_data["时间"] > "2017/03/21 23:45")]
    train_data.reset_index(drop=True,inplace=True)


# ## 2.2 删除实际辐照度明显能够够发电但功率缺位零的样本

# In[11]:


def del_date_0(data):
    print('删除实际辐照度明显能够发电但功率为零的样本')
    print('现有样本数')
    print(data.shape)
    data['年月日']=data['时间'].dt.date
    data['当日时间']=data['时间'].dt.time
    del_data_date_0=data['年月日'][(data['实际功率']==0)&(data['实际辐照度']>600)].unique().tolist()
    for date in del_data_date_0:
        print(date)
    data=data[~data['年月日'].isin(del_data_date_0)]
    print('删除实际应发电日期样本后样本数:',data.shape[0])
    return data


# In[12]:


data=del_date_0(data)


# ## 2.3 删除当日时间内有较多实际功率和辐照度为零的日期样本

# In[13]:


import datetime as dt
def del_date_1(data):
    print('删除有较多实际功率和辐照度为零样本')
    print('现有样本数')
    print(data.shape)
    data['当日时间']=data['时间'].dt.time
    day_time_data=data[(data['当日时间']>dt.time(10,30))&(data['当日时间']<dt.time(15,30))&(data['实际辐照度']==0)]
    df_temp=day_time_data.groupby('年月日')['实际功率'].agg(lambda x: x.value_counts()[0] if 0 in x.values else 0).reset_index()
    del_data_date_1=df_temp[df_temp['实际功率']>5]['年月日'].values.tolist()
    print('删除有较多实际功率和辐照度为零的样本日期：')
    for date in del_data_date_1:
        print(date)
    data=data[~data['年月日'].isin(del_data_date_1)]
    print('删除有较多实际功率和辐照度为零数据后的样本数量：',data.shape[0])
    return data


# In[14]:


data=del_date_1(data)


# ## 2.4 将实际功率/实际辐照度异常的日期数据删除

# In[15]:


def del_date_2(data):
    print('现有样本数')
    print(data.shape)
    temp=data[(data['实际功率']>0)&(data['实际辐照度']>0)]
    temp['实际比例']=temp['实际功率']/temp['实际辐照度']
    df_breakdown_time=temp[(temp['当日时间']>=dt.time(10,00))&(temp['当日时间']<=dt.time(16,00))&(temp['实际比例']<0.005)&(temp['实际辐照度']>600)]
    temp_groupby=df_breakdown_time.groupby('年月日').agg({'实际比例':'count'}).reset_index()
    del_data_date_2=temp_groupby[temp_groupby['实际比例']>5]['年月日'].values.tolist()
    print('实际功率/实际辐照度异常的日期：')
    for date in del_data_date_2:
        print(date)
    data=data[~data['年月日'].isin(del_data_date_2)]
    print('删除有较多实际功率和辐照度为零数据后的样本数量：',data.shape[0])
    return data


# In[16]:


data=del_date_2(data)


# ## 2.5参考正态分布去筛选或填充数据

# In[17]:


## 因为前面有将功率负数更换为0的操作，因此小于或者大于的操作会大大减少数据量


# In[18]:


from copy import deepcopy
import numpy as np
def normal_clean_data(df, index_name, var_name="实际功率",refill=False):
    temp_clean =pd.DataFrame()
    df_train=data[np.isnan(data['实际功率'])==False]
    df_test=data[np.isnan(data['实际功率'])==True]
    for g_name, g in df_train.groupby(index_name):
        temp = deepcopy(g).reset_index(drop=True)
        limit_low, limit_up = np.percentile(temp[var_name], [5, 95])
        if refill:
            temp[var_name][(temp[var_name] >= limit_up)|(temp[var_name] <= limit_low)]=temp[var_name].median()   
        else:
            temp = temp[(temp[var_name] <= limit_up) & (temp[var_name] >= limit_low)].reset_index(drop=True)
        temp_clean=pd.concat([temp_clean,temp])
    print(temp_clean.shape)
    print(df_test.shape)
    df_clean = pd.concat([temp_clean,df_test], ignore_index=True)
    return df_clean


# In[19]:


clean_data=normal_clean_data(data,index_name=['当日时间','month'],refill=False)


# In[20]:


clean_data=clean_data.sort_values(by='时间',ascending=True).reset_index(drop=True)


# ## 2.6 基于密度的离群点判定

# from sklearn.neighbors import LocalOutlierFactor
# clf = LocalOutlierFactor(n_neighbors=1000, contamination=0.1)
# from sklearn.preprocessing import MinMaxScaler
# scaler=MinMaxScaler(feature_range=(0,1))
# sel=data[['实际辐照度','实际功率']][~data['实际功率'].isnull()]
# scaled_sel=scaler.fit_transform(sel)
# error_label=clf.fit_predict(scaled_sel)

# ser=pd.DataFrame(error_label,columns=['是否异常'])
# data=data.reset_index(drop=True)
# data=pd.concat([data,ser],axis=1)
# data=data[data['是否异常']!=-1]

# # 4.特征工程

# ## 4.1 将采用fbprophet预测的时序结果加入特征

# In[21]:


start_time=train_data.iloc[0]['时间']
end_time=train_data.iloc[-1]['时间']
train_selc=clean_data[['时间','实际功率']][(clean_data['时间']>=start_time)&(clean_data['时间']<=end_time)]


# In[22]:


ts=pd.date_range(start_time,end_time,freq='15T')
temp=pd.DataFrame({'ds':ts})
df_full_time=pd.merge(temp,train_selc,how='left',left_on='ds',right_on='时间')
df_full_time.drop('时间',axis=1,inplace=True)
df_full_time.columns=['ds','y']
df_full_time['cap']=cp
df_full_time['floor']=0


# In[23]:


from fbprophet import Prophet
m=Prophet(growth='logistic',daily_seasonality=True,yearly_seasonality=True)
m.fit(df_full_time)


# In[24]:


import datetime as dt
test_start_time=test_data.iloc[0]['时间']
test_end_time=test_data.iloc[-1]['时间']
period=int((test_end_time-test_start_time)/dt.timedelta(minutes=15))+1


# In[25]:


import matplotlib.pyplot as plt
future=m.make_future_dataframe(periods=period,freq='15T',include_history=True)
future['floor']=0
future['cap']=cp
fcst=m.predict(future)
fig=m.plot(fcst)
plt.show()


# In[26]:


fcst_data=fcst[['ds','yhat_lower','yhat','yhat_upper']]


# In[27]:


data=clean_data.merge(fcst_data,how='left',left_on='时间',right_on='ds')


# ## 4.1 将温度等转换为正数区间，否则其乘积无物理意义

# In[28]:


shift_cols=['辐照度','风速','温度','湿度','压强','直辐射','总辐射']
for col in shift_cols:
    data[col]=data[col]+1


# ## 4.2新增小时、月份、day_of_year和sec_of_day作为特征

# def sec_of_day(time):
#     seco,minu,hour=time.second,time.minute,time.hour
#     sec_of_day=(seco+60*minu+60*60*hour)/(24*60*60)
#     return sec_of_day

# In[29]:


data['小时']=data['时间'].apply(lambda x:x.hour)


# ## 4.3 新增总辐射是否为-1的特征

# In[30]:


data['总辐射是否为-1']=data['总辐射'].apply(lambda x: 0 if x==-1 else 1)


# 只要总辐射值为-1，则不计入损失

# data[(data['是否计入损失']!=0) & (data['总辐射']==-1)].shape

# ## 4.4 添加一个辐照度是否小于-0.9的特征

# In[31]:


data['辐照度小于0.9']=data['辐照度'].apply(lambda x:1 if x<-0.9 else 0)


# ## 4.5 将连续的风向值变为分类变量

# In[32]:


风向_max=data['风向'].max()
风向_min=data['风向'].min()
distance=(风向_max-风向_min)/8
data['类别风向']=data['风向'].apply(lambda x:(x-风向_min)/distance)
data['类别风向']=data['类别风向'].round().astype(int)


# ## 4.6 定义一个是否小于电站输出阈值的列

# In[33]:


## 注意是不是对训练集数据有影响
limit=cp*0.03
data['是否小于阈值']=train_data['实际功率'].apply(lambda x:1 if x<limit else 0)


# ## 4.7 添加一个与当月平均值差值的特征

# In[34]:


def mon_mean_delta(info_word,df):
    merge_data=df.groupby(['month','小时'])[info_word].agg('mean').reset_index()
    merge_data.rename(columns={info_word:'{}_month_mean'.format(info_word)},inplace=True)
    df=df.merge(merge_data,how='left',left_on=['month','小时'],right_on=['month','小时'],sort=False)
    df['{}_mean_delta'.format(info_word)]=df['{}_month_mean'.format(info_word)]-df[info_word]
    return df


# In[35]:


for feature in ['温度','湿度','辐照度','风速','压强','风向']:
    data=mon_mean_delta(feature,data)


# ## 4.8 增加一个季节特征

# In[36]:


import math
data['season']=data['month'].apply(lambda x: math.floor((x)/3) if x!=12 else 0)


# ## 4.9 增加与上一时刻值特征

# In[37]:


data_shift=data.shift(1).reset_index(drop=True)


# In[38]:


for feature in ['温度','湿度','辐照度','风速','压强','风向','总辐射是否为-1','时间']:
    data['{}_last'.format(feature)]=data_shift[feature]


# ## 4.10添加上一时刻是否为断点特征

# In[39]:


data=data.drop([0],axis=0).reset_index()


# In[40]:


import numpy as np
data['delta_time']=data['时间']-data['时间_last']
data['是否为断点']=data['delta_time'].apply(lambda x:0 if x==np.timedelta64(15,'m') else 1)


# ## 4.11 增加与上一时刻值变化特征

# In[41]:


for feature in ['温度','湿度','辐照度','风速','压强','风向']:
    data['{}_last_delta'.format(feature)]=(data[feature]-data['{}_last'.format(feature)])


# # 5 将处理后的数据保存到文件中

# In[42]:


del data['index']


# In[43]:


data.to_csv(file_path+'final_data/final_data'+str(num)+'.csv',encoding='gbk',index=False)


# In[44]:


print('\n \n')

