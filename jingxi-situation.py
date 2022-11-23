#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install peakutils')
get_ipython().system('pip install neurokit2')
#加载所需要用到的库
import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.options.plotting.backend = "plotly"
import seaborn as sns
import scipy.signal as signal
import pywt
import wfdb
import peakutils
import neurokit2 as nk


# # Loading data

# ## jingxi.mat

# In[2]:


dataFile_jingxi = '../input/jingxi-ecg/jingxi.mat'
data_jingxi = scio.loadmat(dataFile_jingxi)
all_data_jingxi = data_jingxi['data']
ECG_jingxi_data_3 = all_data_jingxi[:,1]


# # ECG-jingxi

# In[3]:


#可视化
ecg_jingxi_df_3 = pd.DataFrame({"ECG_jingxi_Ⅲ": ECG_jingxi_data_3})
ecg_jingxi_df_3.plot(title="ECG_jingxi_Ⅲ",labels=dict(index="Time", value="Amp"))


# In[4]:


ECG_jingxi_data_2 = all_data_jingxi[:,5]
#可视化
ecg_jingxi_df_2 = pd.DataFrame({"ECG_jingxi_Ⅱ": ECG_jingxi_data_2})
ecg_jingxi_df_2.plot(title="ECG_jingxi_Ⅱ",labels=dict(index="Time", value="Amp"))


# In[5]:


ECG_jingxi_data_1 = all_data_jingxi[:,1]-all_data_jingxi[:,5]
#可视化
ecg_jingxi_df_1 = pd.DataFrame({"ECG_jingxi_Ⅰ": ECG_jingxi_data_1})
ecg_jingxi_df_1.plot(title="ECG_jingxi_Ⅰ",labels=dict(index="Time", value="Amp"))


# # PPG-jingxi

# In[6]:


PPG_jingxi_data = all_data_jingxi[:,0]
#可视化
PPG_jingxi = pd.DataFrame({"PPG_jingxi_data": PPG_jingxi_data})
PPG_jingxi.plot(title="PPG_jingxi_data",labels=dict(index="Time", value="脉搏波"))


# # Processing PPG-jingxi

# In[7]:


ppg_signals, ppg_info = nk.ppg_process(PPG_jingxi_data)
nk.ppg_plot(ppg_signals)
cleaned_ppg = ppg_signals["PPG_Clean"]
cleaned_ppg_rate = ppg_signals["PPG_Rate"]


# In[8]:


ppg_raw_cleaned = pd.DataFrame({"PPG_jingxi_data": PPG_jingxi_data,"cleaned_ppg": cleaned_ppg})
ppg_raw_cleaned.plot(title="PPG_Raw_and_Cleaned",labels=dict(index="Time", value="脉搏波"))


# In[9]:


cleaned_ppg_rate.plot(title="PPG_Rate",labels=dict(index="Time", value="次数"))


# # Processing ECG-Ⅱ

# In[10]:


# 默认处理管道
signals, info = nk.ecg_process(ECG_jingxi_data_2, sampling_rate=1000)
#0.5 Hz high-pass butterworth filter (order = 5), followed by powerline filtering (see signal_filter()). By default, powerline = 50.
#可视化
nk.ecg_plot(signals[0:10000])
cleaned_ecg_2 = signals["ECG_Clean"]
cleaned_ecg_rate = signals["ECG_Rate"]


# In[11]:


ecg_2_raw_cleaned = pd.DataFrame({"ECG_jingxi_Ⅱ": ECG_jingxi_data_2,"ECG_cleaned_Ⅱ": cleaned_ecg_2})
ecg_2_raw_cleaned.plot(title="Raw_and_Cleaned",labels=dict(index="Time", value="Amp"))


# # Find the ECG-jingxi-Ⅱ R peaks

# In[12]:


import plotly.graph_objs as go
import plotly.offline as py
from plotly import tools
# Extract R-peaks.
ecg_2_R_Peaks = np.where(signals["ECG_R_Peaks"] == 1)[0]
trace1 = go.Scatter(
                    x = ecg_2_R_Peaks,   
                    y = cleaned_ecg_2[ecg_2_R_Peaks],
                    mode = "markers",
                    name= 'ecg_2_R_Peaks')
trace2 = go.Scatter(
                    x = np.arange(len(cleaned_ecg_2)),
                    y = cleaned_ecg_2,
                    mode = "lines",
                    name= 'cleaned_ecg_2')
# 添加图层layout
layout = dict(title = 'Find the ECG-jingxi-Ⅱ R peaks',
              # 设置图像的标题
              xaxis= dict(title= 'Time'),
              # 设置x轴名称，x轴刻度线的长度，不显示零线
              yaxis= dict(title= 'Amp'),
              # 设置y轴名称，x轴刻度线的长度，不显示零线
             ) 
data = [trace1,trace2]
fig = dict(data = data,layout = layout)
py.iplot(fig)


# # Find the PPG-jingxi R peaks

# In[13]:


import plotly.graph_objs as go
import plotly.offline as py
from plotly import tools

ppg_R_Peaks = np.where(ppg_signals["PPG_Peaks"] == 1)[0]
trace1 = go.Scatter(
                    x = ppg_R_Peaks,   
                    y = cleaned_ppg[ppg_R_Peaks],
                    mode = "markers",
                    name= 'PPG_Peaks')
trace2 = go.Scatter(
                    x = np.arange(len(cleaned_ppg)),
                    y = cleaned_ppg,
                    mode = "lines",
                    name= 'cleaned_PPG')
# 添加图层layout
layout1 = dict(title = 'Find the PPG-jingxi R peaks',
              # 设置图像的标题
              xaxis= dict(title= 'Time'),
              # 设置x轴名称，x轴刻度线的长度，不显示零线
              yaxis= dict(title= '脉搏波'),
              # 设置y轴名称，x轴刻度线的长度，不显示零线
             ) 
data1 = [trace1,trace2]
fig = dict(data = data1,layout = layout1)
py.iplot(fig)


# # Rate

# In[14]:


cleaned_ecg_rate.plot(title="Ecg_jingxi_Ⅱ_Rate",labels=dict(index="Time", value="次数"))


# In[15]:


qrs_epochs = nk.ecg_segment(cleaned_ecg_2, rpeaks=None, show=True)


# # HRV

# ## Time-Domain Analysis

# In[16]:


# Extract clean EDA and SCR features
hrv_time = nk.hrv_time(ecg_2_R_Peaks, show=True)
hrv_time


# ## Frequency-Domain Analysis

# In[17]:


hrv_freq = nk.hrv_frequency(ecg_2_R_Peaks, show=True, normalize=True)
hrv_freq


# ## Non-Linear Domain Analysis

# In[ ]:


hrv_nonlinear = nk.hrv_nonlinear(ecg_2_R_Peaks, show=True)
hrv_nonlinear


# # Heartbeat

# In[26]:


qrs_epochs = nk.ecg_segment(cleaned_ecg_2, show = True)


# In[23]:


np.mean(cleaned_ecg_rate)


# In[24]:


np.std(cleaned_ecg_rate)

