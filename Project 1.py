
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
# import scipy.stat as sps
import matplotlib.pyplot as plt
import seaborn as sns
# from pandas import ExcelWriter
import os
cwd = os.getcwd()
cwd


# In[2]:


get_ipython().run_cell_magic('time', '', "df = pd.read_csv('C:\\\\Users\\\\beizh\\\\Desktop\\\\fraud analytics\\\\Class 1\\\\HW\\\\NY property csv.csv')")


# In[3]:


df.head()


# In[4]:


# Create new variable BORO
df['BORO'] = ''
df['BORO'] = df.BBLE.str[0]

# sns.distplot(df[df['BORO'] == '5']['LTFRONT'],bins = 1000, kde=False)
# plt.xlim(0,500)


# In[5]:


# ZIP, group by BORO, BLOCK
# df['ZIP'] = df.groupby(['BBLE'])['ZIP'].transform(lambda x: x.fillna(x.mean())).astype('int')
# BBLE


# In[179]:


np.std(df['LTFRONT'])
# std = 73.7 without group
# two methods to set 0 values as missing values
df.loc[df['LTFRONT'] == 0, 'LTFRONT'] = np.nan
# df['LTFRONT'] = df['LTFRONT'].replace(0,np.nan)

# fill NA with average values grouped by BORO (if using TAXCLASS as well, the filled values will be too big and there will be 2 missing values)
df['LTFRONT'] = df.groupby(['BORO'])['LTFRONT'].transform(lambda x: x.fillna(x.mean()))

np.isnan(df['LTFRONT']).sum()


# In[180]:


df.loc[df['LTDEPTH'] == 0, 'LTDEPTH'] = np.nan
df['LTDEPTH'] = df.groupby(['BORO'])['LTDEPTH'].transform(lambda x: x.fillna(x.mean()))
np.isnan(df['LTDEPTH']).sum()


# In[181]:


# Calculate BLDFRONT/BLDDEPTH ratio
df.loc[df['BLDFRONT'] == 0, 'BLDFRONT'] = np.nan
df.loc[df['BLDDEPTH'] == 0, 'BLDDEPTH'] = np.nan

df['bb_rate'] = df['BLDFRONT']/df['BLDDEPTH']
# distribution of the ratio
# sns.distplot(df['BLDFRONT']/df['BLDDEPTH'].dropna(), bins = 1000,kde=False)
# plt.xlim(0,2)

median_bbrate = np.median(df['bb_rate'].dropna())
df['BLDFRONT'] = df['BLDFRONT'].fillna(value = (df['BLDDEPTH'] * median_bbrate))
df['BLDDEPTH'] = df['BLDDEPTH'].fillna(value = (df['BLDFRONT'] / median_bbrate))


# In[189]:


# For BLDFRONT and BLDDEPTH, if group by both BORO and TAXCLASS, there will appear NaN as well.
df['BLDFRONT'] = df.groupby(['BORO'])['BLDFRONT'].transform(lambda x: x.fillna(x.mean()))

# np.isnan(df['BLDFRONT']).sum()
df['BLDDEPTH'] = df.groupby(['BORO'])['BLDDEPTH'].transform(lambda x: x.fillna(x.mean()))


# In[193]:


# STORIES, group by TAXCLASS
df['STORIES'] = df.groupby(['TAXCLASS'])['STORIES'].transform(lambda x: x.fillna(x.mean()))


# In[194]:


# building volume and bins accordingly
df['bldvol'] = df['BLDFRONT'] * df['BLDDEPTH'] * df['STORIES']

# xhigh = 60000
# plt.xlim(0,xhigh)
# temp = df[df['bldvol'] <= xhigh]
# sns.distplot(temp['bldvol'],bins=200, kde=False)

df['bldvol_bin'] = pd.qcut(df['bldvol'], 100, labels = False, duplicates = 'drop')


# In[200]:


# FULLVAL
df.loc[df['FULLVAL'] == 0, 'FULLVAL'] = np.nan

# df['FULLVAL']
df['FULLVAL'] = df.groupby(['BORO','bldvol_bin'])['FULLVAL'].transform(lambda x: x.fillna(x.mean()))
np.isnan(df['FULLVAL']).sum()


# In[198]:


# AVLAND
df.loc[df['AVLAND'] == 0, 'AVLAND'] = np.nan
df['AVLAND'] = df.groupby(['BORO','bldvol_bin'])['AVLAND'].transform(lambda x: x.fillna(x.mean()))
np.isnan(df['AVLAND']).sum()


# In[199]:


# AVTOT
df.loc[df['AVTOT'] == 0, 'AVTOT'] = np.nan
df['AVTOT'] = df.groupby(['BORO','bldvol_bin'])['AVTOT'].transform(lambda x: x.fillna(x.mean()))
np.isnan(df['AVTOT']).sum()


# In[201]:


df['lotarea'] = df['LTDEPTH'] * df['LTFRONT']
df['bldarea'] = df['BLDFRONT'] * df['BLDDEPTH']

