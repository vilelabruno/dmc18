from __future__ import division

print 'Loading libraries...'

import pandas as pd 
import numpy as np 
from xgboost.sklearn import XGBRegressor
import os
import time
import datetime
import matplotlib.pyplot as pyplot
from sklearn.ensemble import RandomForestClassifier

print 'Setting random seed...'
seed = 1234
np.random.seed(seed)

#%%
print 'Reading csv...'
#%%
train = pd.read_csv('train.csv', sep=',')
#items = pd.read_csv('items.csv', sep='|')
prices = pd.read_csv('newprices.csv', sep=',')
test = pd.read_csv('test.csv', sep=',')
s = pd.read_csv('new_stock.csv')
train = train.fillna(-1)
test = test.fillna(-1)
print 'Treating variables...'
#date, releaseDate
train['date'] = pd.to_datetime(train['date'])
train['releaseDate'] = pd.to_datetime(train['releaseDate'])
test['releaseDate'] = pd.to_datetime(test['releaseDate'])

#TODO implement a best way to pass size to ml

train['size'] = train['pid'][train['pid'].str.contains('_') == True].str.split('_').str[1]
test['size'] = test['pid'][test['pid'].str.contains('_') == True].str.split('_').str[1]

train['mainCategory'] = train['mainCategory'].astype(str)
train['category'] = train['category'].astype(str)
train['size'] = train['size'].astype(str)

train['size'] = train[['mainCategory', 'size']].apply('_'.join, axis=1)

test['mainCategory'] = test['mainCategory'].astype(str)
test['category'] = test['category'].astype(str)
test['size'] = test['size'].astype(str)
test['size'] = test[['mainCategory', 'size']].apply('_'.join, axis=1)

count_size = train.groupby('size')['category'].count().reset_index()
count_size.columns = ['size', 'count_size']
train = pd.merge(train, count_size, on='size', how='left', sort=False)
train.drop('size', axis=1, inplace=True)

count_size = test.groupby('size')['category'].count().reset_index()
count_size.columns = ['size', 'count_size']
test = pd.merge(test, count_size, on='size', how='left', sort=False)
test.drop('size', axis=1, inplace=True)

#
#
#count_size = train.groupby('size')['mainCategory'].count().reset_index()
#count_size.columns = ['size', 'count_size']
#train = pd.merge(train, count_size, on='size', how='left', sort=False)
#train.drop('size', axis=1, inplace=True)
#
#count_size = test.groupby('size')['mainCategory'].count().reset_index()
#count_size.columns = ['size', 'count_size']
#test = pd.merge(test, count_size, on='size', how='left', sort=False)
#test.drop('size', axis=1, inplace=True)
#
#
#print train['count_size'].value_counts()

#brand with others brand (not good)

#train['brand'][train['brand'] == 'Lotto'] = "others_brand"
#train['brand'][train['brand'] == 'Stance'] = "others_brand"
#train['brand'][train['brand'] == 'Reusch'] = "others_brand"
#train['brand'][train['brand'] == 'Cinquestelle'] = "others_brand"
#train['brand'][train['brand'] == 'Diadora'] = "others_brand"
#train['brand'][train['brand'] == 'Under Armour'] = "others_brand"
#
#test['brand'][test['brand'] == 'Lotto'] = "others_brand"
#test['brand'][test['brand'] == 'Stance'] = "others_brand"
#test['brand'][test['brand'] == 'Reusch'] = "others_brand"
#test['brand'][test['brand'] == 'Cinquestelle'] = "others_brand"
#test['brand'][test['brand'] == 'Diadora'] = "others_brand"
#test['brand'][test['brand'] == 'Under Armour'] = "others_brand"



train.drop('brand', axis=1, inplace=True)
train.drop('color', axis=1, inplace=True)
test.drop('brand', axis=1, inplace=True)
test.drop('color', axis=1, inplace=True)
print 'Merging DataFrames...'
train_pr = pd.merge(train, prices, how='inner', left_on='pid', right_on='pid')
#test_pr = pd.merge(test, prices, how='inner', left_on='pid', right_on='pid')

print 'Separating train in 3 months...'
mask = train['date']

octo = train.where((mask >= pd.Timestamp('2017-10-01')) & (mask < pd.Timestamp('2017-11-01'))).dropna()
nove = train.where((mask >= pd.Timestamp('2017-11-01')) & (mask < pd.Timestamp('2017-12-01'))).dropna()
dece = train.where((mask >= pd.Timestamp('2017-12-01')) & (mask <= pd.Timestamp('2017-12-31'))).dropna()

print 'Saving last selling date of each product in each month...'
gpo = octo.groupby('pid').agg({'date': np.max, 'units': np.sum}).reset_index()
gpn = nove.groupby('pid').agg({'date': np.max, 'units': np.sum}).reset_index()
gpd = dece.groupby('pid').agg({'date': np.max, 'units': np.sum}).reset_index()
print gpo

gpo = pd.merge(gpo, octo, on=['pid', 'date'], how='left')
gpn = pd.merge(gpn, nove, on=['pid', 'date'], how='left')
gpd = pd.merge(gpd, dece, on=['pid', 'date'], how='left')

print gpo[gpo['date'] < pd.Timestamp('2017-10-20')].describe()
print gpo[gpo['date'] > pd.Timestamp('2017-10-20')].describe()
print 'Dropping pid...'
octo_pid = gpo['pid']
nove_pid = gpn['pid']
dece_pid = gpd['pid']
test_pid = test['pid']
gpo.drop('pid', axis=1, inplace=True)
gpn.drop('pid', axis=1, inplace=True)
gpd.drop('pid', axis=1, inplace=True)
test.drop('pid', axis=1, inplace=True)
test.drop('units', axis=1, inplace=True)

print 'Dropping target...'
octo_target = gpo['date']
nove_target = gpn['date']
dece_target = gpd['date']
gpo.drop('date', axis=1, inplace=True)
gpn.drop('date', axis=1, inplace=True)
gpd.drop('date', axis=1, inplace=True)

print 'Changing datetime to float...'
octo_target = octo_target.apply(lambda x: x.strftime('%d'))
nove_target = nove_target.apply(lambda x: x.strftime('%d'))
dece_target = dece_target.apply(lambda x: x.strftime('%d'))



octo_target = octo_target.astype(int)
nove_target = nove_target.astype(int)
dece_target = dece_target.astype(int)


print 'Instantiating xgboost model...'
clf = XGBRegressor(  learning_rate=          0.2,
                     n_estimators=           100,
                     max_depth=              5,
                     subsample=              1.0,
                     colsample_bytree=       1.0,
                     objective=              'reg:linear',
                     nthread=                3,
                     seed=                   seed)

#december nao levar em consideracao zdezembro
print 'Training xgboost model...'
gpfinal = pd.concat([gpo, gpn])
final_target =  pd.concat([octo_target, nove_target]) 

print len(gpfinal)


gpfinal['stock'] = gpfinal.units_x.apply(lambda x: x**4)
test['stock'] = s['units'].apply(lambda x: x**4)
gpfinal['stock2'] = gpfinal['stock']
gpfinal['stock3'] = gpfinal['stock']
gpfinal['stock4'] = gpfinal['stock']
gpfinal['stock5'] = gpfinal['stock']
test['stock2'] = test['stock']
test['stock3'] = test['stock']
test['stock4'] = test['stock']
test['stock5'] = test['stock']
print test.stock
gpfinal.drop('units_x', axis=1, inplace=True)
gpfinal.drop('units_y', axis=1, inplace=True)

gpfinal['mainCategory'] = gpfinal['mainCategory'].astype(int)  
test['mainCategory'] = test['mainCategory'].astype(int)
gpfinal['category'] = gpfinal['category'].astype(int)  
test['category'] = test['category'].astype(int)
gpfinal.drop('subCategory', axis=1, inplace=True)
test.drop('subCategory', axis=1, inplace=True)
gpfinal.drop('mainCategory', axis=1, inplace=True)
test.drop('mainCategory', axis=1, inplace=True)
gpfinal.drop('category', axis=1, inplace=True)
test.drop('category', axis=1, inplace=True)
gpfinal.drop('rrp', axis=1, inplace=True)
test.drop('rrp', axis=1, inplace=True)
gpfinal.drop('releaseDate', axis=1, inplace=True)
test.drop('releaseDate', axis=1, inplace=True)

clf.fit(gpfinal, final_target, eval_metric='mae',
        eval_set=[(gpfinal, final_target)])

print clf.booster().get_score()

preds = clf.predict(test)

clf2 = RandomForestClassifier(n_jobs=2, random_state=0)
clf2.fit(gpfinal, final_target)
preds2 = clf2.predict(test)

#from sklearn.model_selection import train_test_split
#X_train, X_validation, y_train, y_validation = train_test_split(gpfinal, final_target, train_size=0.7, random_state=seed)
#categorical_features_indices = np.where(gpfinal.dtypes != np.float)[0]
#from catboost import CatBoostRegressor
#model=CatBoostRegressor(iterations=100, depth=3, learning_rate=0.1, loss_function='RMSE')
#model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_validation, y_validation),plot=True)
#
#preds3 = model.predict(test)
#preds = (np.array(preds) + np.array(preds2) + np.array(preds3)) / 3
preds = (np.array(preds) + np.array(preds2)) / 2

#print 'Converting timestamp to datetime...'
#preds = preds.apply(lambda x: datetime.datetime.fromtimestamp(int(x)).strftime('%Y-%m-%d'))

print 'Writing submission_file...'
sub_file = pd.read_csv('sampleSubmission.csv')

sub_file['Day'] = preds.astype(int)
sub_file['Day'][sub_file['Day'] < 30] += 2
print sub_file.Day
sub_file.to_csv('submission_003.csv', index=False)

print(preds.dtype)

print 'Done!'