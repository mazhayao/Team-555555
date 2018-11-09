 # ----------- Overhead -------------

import pandas as pd
import numpy as np
# import scipy.stat as sps
import matplotlib.pyplot as plt
import seaborn as sns
# from pandas import ExcelWriter

df = pd.read_csv('NY property csv.csv')



# --------------- Fill in NAs and 0s ----------

# Create new variable BORO
df['BORO'] = ''
df['BORO'] = df.BBLE.str[0]

# ZIP, sort by BBLE and forward fill
df = df.sort_values(by = ['BBLE'])
df['ZIP'].fillna(method='ffill',inplace = True)


df['ll_rate'] = df['LTFRONT']/df['LTDEPTH']
mean_llrate = np.mean(df['ll_rate'].dropna())
df['LTFRONT'] = df['LTFRONT'].fillna(value = (df['LTDEPTH'] * mean_llrate))
df['LTFRONT'] = df['LTDEPTH'].fillna(value = (df['LTFRONT'] / mean_llrate))
np.std(df['LTFRONT'])
# std = 73.7 without group
# two methods to set 0 values as missing values
df.loc[df['LTFRONT'] == 0, 'LTFRONT'] = np.nan
# df['LTFRONT'] = df['LTFRONT'].replace(0,np.nan)

# fill NA with average values grouped by BORO (if using TAXCLASS as well, the filled values will be too big and there will be 2 missing values)
df['LTFRONT'] = df.groupby(['BORO','TAXCLASS'])['LTFRONT'].transform(lambda x: x.fillna(x.mean()))
df['LTFRONT'] = df.groupby(['TAXCLASS'])['LTFRONT'].transform(lambda x: x.fillna(x.mean()))


df.loc[df['LTDEPTH'] == 0, 'LTDEPTH'] = np.nan
df['LTDEPTH'] = df.groupby(['BORO','TAXCLASS'])['LTDEPTH'].transform(lambda x: x.fillna(x.mean()))
df['LTDEPTH'] = df.groupby(['TAXCLASS'])['LTDEPTH'].transform(lambda x: x.fillna(x.mean()))

# np.isnan(df['LTDEPTH']).sum()

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


# For BLDFRONT and BLDDEPTH, if group by both BORO and TAXCLASS, there will appear NaN as well.
df['BLDFRONT'] = df.groupby(['BORO','TAXCLASS'])['BLDFRONT'].transform(lambda x: x.fillna(x.mean()))
df['BLDFRONT'] = df.groupby(['TAXCLASS'])['BLDFRONT'].transform(lambda x: x.fillna(x.mean()))

# np.isnan(df['BLDFRONT']).sum()
df['BLDDEPTH'] = df.groupby(['BORO','TAXCLASS'])['BLDDEPTH'].transform(lambda x: x.fillna(x.mean()))
df['BLDDEPTH'] = df.groupby(['TAXCLASS'])['BLDDEPTH'].transform(lambda x: x.fillna(x.mean()))

np.isnan(df['BLDFRONT']).sum()
np.isnan(df['BLDDEPTH']).sum()

df['STORIES'] = df.groupby(['ZIP','TAXCLASS'])['STORIES'].transform(lambda x: x.fillna(x.mean()))
df['STORIES'] = df.groupby(['TAXCLASS'])['STORIES'].transform(lambda x: x.fillna(x.mean()))

# np.isnan(df['STORIES']).sum()

# building volume and bins accordingly
df['bldvol'] = df['BLDFRONT'] * df['BLDDEPTH'] * df['STORIES']

df['bldvol_bin'] = pd.qcut(df['bldvol'], 100, labels = False, duplicates = 'drop')

# FULLVAL
df.loc[df['FULLVAL'] == 0, 'FULLVAL'] = np.nan

# df['FULLVAL']
df['FULLVAL'] = df.groupby(['BORO','bldvol_bin'])['FULLVAL'].transform(lambda x: x.fillna(x.mean()))
np.isnan(df['FULLVAL']).sum()


# AVLAND
df.loc[df['AVLAND'] == 0, 'AVLAND'] = np.nan
df['AVLAND'] = df.groupby(['BORO','bldvol_bin'])['AVLAND'].transform(lambda x: x.fillna(x.mean()))
np.isnan(df['AVLAND']).sum()


# AVTOT
df.loc[df['AVTOT'] == 0, 'AVTOT'] = np.nan
df['AVTOT'] = df.groupby(['BORO','bldvol_bin'])['AVTOT'].transform(lambda x: x.fillna(x.mean()))
np.isnan(df['AVTOT']).sum()


## Build 3 sizes
## Build LotArea, Building Area, Building Volume
df['lotarea'] = df['LTDEPTH'] * df['LTFRONT']
df['bldarea'] = df['BLDFRONT'] * df['BLDDEPTH']
df['bldvol'] = df['bldarea'] * df['STORIES']

## Build 9 values
## FV = FULLVAL, AL = AVLAND, AT = AVTOT
## LA = LOTAREA, BA = BLDAREA, BV = BLDVOL
df['fv_la'] = df['FULLVAL'] / df['lotarea']
df['fv_ba'] = df['FULLVAL'] / df['bldarea']
df['fv_bv'] = df['FULLVAL'] / df['bldvol']
df['al_la'] = df['AVLAND'] / df['lotarea']
df['al_ba'] = df['AVLAND'] / df['bldarea']
df['al_bv'] = df['AVLAND'] / df['bldvol']
df['at_la'] = df['AVTOT'] / df['lotarea']
df['at_ba'] = df['AVTOT'] / df['bldarea']
df['at_bv'] = df['AVTOT'] / df['bldvol']

## Build 45 Group by values
ninevars = ['fv_la','fv_ba','fv_bv','al_la','al_ba','al_bv','at_la','at_ba','at_bv']
df = df.join(df.groupby(['ZIP'])[ninevars].mean(), on='ZIP', rsuffix='_zip')
df = df.join(df.groupby(['ZIP3'])[ninevars].mean(), on='ZIP3', rsuffix='_zip3')
df = df.join(df.groupby(['TAXCLASS'])[ninevars].mean(), on='TAXCLASS', rsuffix='_taxclass')
df = df.join(df.groupby(['BORO'])[ninevars].mean(), on='BORO', rsuffix='_boro')


