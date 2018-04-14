import pandas as pd
import numpy as np

train = pd.read_csv('train.csv', sep="|")
items = pd.read_csv('items.csv', sep="|")
train = train.fillna(-1)
mask = train['date'].apply(lambda x: pd.Timestamp(x))

jan = train.where((mask >= pd.Timestamp('2018-01-01')) & (mask < pd.Timestamp('2018-02-01'))).dropna()
jan['pid'] = jan['pid'].astype(int)
jan['pid'] = jan['pid'].astype(str)
jan['size'] = jan['size'].astype(str)
jan['pid'] = jan[['pid','size']].apply('_'.join, axis=1)
items['pid'] = items['pid'].astype(int)
items['pid'] = items['pid'].astype(str)
items['size'] = items['size'].astype(str)
items['pid'] = items[['pid','size']].apply('_'.join, axis=1)



jan = jan.groupby('pid').agg({'units': np.sum, 'date': np.max}).reset_index()
aux = pd.merge(jan.where((jan.date.apply(lambda x: pd.Timestamp(x)) >= pd.Timestamp('2018-01-01')) & (jan.date.apply(lambda x: pd.Timestamp(x)) < pd.Timestamp('2018-01-07'))).dropna(), items, on="pid", how='left')

print aux

aux = pd.merge(jan.where((jan.date.apply(lambda x: pd.Timestamp(x)) >= pd.Timestamp('2018-01-15')) & (jan.date.apply(lambda x: pd.Timestamp(x)) < pd.Timestamp('2018-01-28'))).dropna(), items, on="pid", how='left')
print aux
#train[train.isnull().any(axis=1)] get na values