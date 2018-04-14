from __future__ import division

print 'Loading libraries...'

import pandas as pd 
import numpy as np 
from xgboost.sklearn import XGBRegressor
import os
import time
import datetime

print 'Setting random seed...'
seed = 1234
np.random.seed(seed)

print 'Reading csv...'
train = pd.read_csv('train.csv', sep=',')
#items = pd.read_csv('items.csv', sep='|')
prices = pd.read_csv('newprices.csv', sep=',')
test = pd.read_csv('test.csv', sep=',')

print 'Treating variables...'
#date, releaseDate
train['date'] = pd.to_datetime(train['date'])
train['releaseDate'] = pd.to_datetime(train['releaseDate'])
test['releaseDate'] = pd.to_datetime(test['releaseDate'])

octo = train.where((train['releaseDate'] > pd.Timestamp('2017-10-01')) & (train['releaseDate'] < pd.Timestamp('2017-11-01'))).dropna()
nove = train.where((train['releaseDate'] >= pd.Timestamp('2017-11-01')) & (train['releaseDate'] < pd.Timestamp('2017-12-01'))).dropna()
dece = train.where((mask >= pd.Timestamp('2017-12-01')) & (mask <= pd.Timestamp('2017-12-31'))).dropna()

for vc, vr in train.groupby('brand'):
    train[vc] = train.brand[train.brand == vc]
    train[vc][train.brand == vc] = 1
    train[vc] = train[vc].fillna(-1)

for vc, vr in train.groupby('color'):
    train[vc] = train.brand[train.brand == vc]
    train[vc][train.brand == vc] = 1
    train[vc] = train[vc].fillna(-1)

for vc, vr in test.groupby('brand'):
    test[vc] = test.brand[test.brand == vc]
    test[vc][test.brand == vc] = 1
    test[vc] = test[vc].fillna(-1)

for vc, vr in test.groupby('color'):
    test[vc] = test.brand[test.brand == vc]
    test[vc][test.brand == vc] = 1
    test[vc] = test[vc].fillna(-1)

train.drop('brand', axis=1, inplace=True)
train.drop('color', axis=1, inplace=True)
train.drop('Onitsuka', axis=1, inplace=True)
train.drop('FREAM', axis=1, inplace=True)
train.drop('Kempa', axis=1, inplace=True)
train.drop('Sells', axis=1, inplace=True)
test.drop('brand', axis=1, inplace=True)
test.drop('color', axis=1, inplace=True)
print 'Merging DataFrames...'
train_pr = pd.merge(train, prices, how='inner', left_on='pid', right_on='pid')
test_pr = pd.merge(test, prices, how='inner', left_on='pid', right_on='pid')

print 'Separating train in 3 months...'
mask = train['date']

octo = train.where((mask >= pd.Timestamp('2017-10-01')) & (mask < pd.Timestamp('2017-11-01'))).dropna()
nove = train.where((mask >= pd.Timestamp('2017-11-01')) & (mask < pd.Timestamp('2017-12-01'))).dropna()
dece = train.where((mask >= pd.Timestamp('2017-12-01')) & (mask <= pd.Timestamp('2017-12-31'))).dropna()

print 'Saving last selling date of each product in each month...'
gpo = octo.groupby('pid').agg({'date': np.max}).reset_index()
gpn = nove.groupby('pid').agg({'date': np.max}).reset_index()
gpd = dece.groupby('pid').agg({'date': np.max}).reset_index()


gpo = pd.merge(gpo, octo, on=['pid', 'date'], how='left')
gpn = pd.merge(gpn, nove, on=['pid', 'date'], how='left')
gpd = pd.merge(gpd, dece, on=['pid', 'date'], how='left')

print 'Dropping pid...'
octo_pid = gpo['pid']
nove_pid = gpn['pid']
dece_pid = gpd['pid']
test_pid = test['pid']
gpo.drop('pid', axis=1, inplace=True)
gpn.drop('pid', axis=1, inplace=True)
gpd.drop('pid', axis=1, inplace=True)
test.drop('pid', axis=1, inplace=True)

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

gpo['releaseDate'] = gpo['releaseDate'].apply(lambda x: x.strftime('%Y-%m-%d'))
gpn['releaseDate'] = gpn['releaseDate'].apply(lambda x: x.strftime('%Y-%m-%d'))
gpd['releaseDate'] = gpd['releaseDate'].apply(lambda x: x.strftime('%Y-%m-%d'))
test['releaseDate'] = test['releaseDate'].apply(lambda x: x.strftime('%Y-%m-%d'))

gpo['releaseDate'] = gpo['releaseDate'].apply(lambda x: time.mktime(datetime.datetime.strptime(x, "%Y-%m-%d").timetuple()))
gpn['releaseDate'] = gpn['releaseDate'].apply(lambda x: time.mktime(datetime.datetime.strptime(x, "%Y-%m-%d").timetuple()))
gpd['releaseDate'] = gpd['releaseDate'].apply(lambda x: time.mktime(datetime.datetime.strptime(x, "%Y-%m-%d").timetuple()))
test['releaseDate'] = test['releaseDate'].apply(lambda x: time.mktime(datetime.datetime.strptime(x, "%Y-%m-%d").timetuple()))

print 'Instantiating xgboost model...'
clf = XGBRegressor(  learning_rate=          0.2,
                     n_estimators=           100,
                     max_depth=              5,
                     subsample=              1.0,
                     colsample_bytree=       1.0,
                     objective=              'multi:softmax',
                     nthread=                3,
                     seed=                   seed)

#december
print 'Training xgboost model...'
gpfinal = pd.concat([gpo, gpn, gpd])
final_target =  pd.concat([octo_target, nove_target, dece_target]) 

clf.fit(gpfinal, final_target, eval_metric='mae',
        eval_set=[(gpfinal, final_target)])

print 'Predicting on test set...'
preds = clf.predict(test)

#print 'Converting timestamp to datetime...'
#preds = preds.apply(lambda x: datetime.datetime.fromtimestamp(int(x)).strftime('%Y-%m-%d'))

print 'Writing submission_file...'
sub_file = pd.read_csv('sampleSubmission.csv')

sub_file['Day'] = preds.astype(int)
sub_file.to_csv('submission_003.csv', index=False)

print(preds.dtype)

print 'Done!'