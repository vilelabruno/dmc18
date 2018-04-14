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

print 'Saving last selling date of each product in each month...'
gpfinal = train.groupby('pid').agg({'date': np.max, 'units': np.sum}).reset_index()

gpfinal = pd.merge(gpfinal, train, on=['pid', 'date'], how='left')

print 'Dropping pid...'
test_pid = test['pid']
gpfinal.drop('pid', axis=1, inplace=True)
test.drop('pid', axis=1, inplace=True)

test.drop('units', axis=1, inplace=True)
print 'Dropping target...'
final_target = gpfinal['date']
gpfinal.drop('date', axis=1, inplace=True)

print 'Changing datetime to float...'
final_target = final_target.apply(lambda x: x.strftime('%d'))

final_target = final_target.astype(int)

gpfinal['releaseDate'] = gpfinal['releaseDate'].apply(lambda x: x.strftime('%Y-%m-%d'))
test['releaseDate'] = test['releaseDate'].apply(lambda x: x.strftime('%Y-%m-%d'))

gpfinal['releaseDate'] = gpfinal['releaseDate'].apply(lambda x: time.mktime(datetime.datetime.strptime(x, "%Y-%m-%d").timetuple()))
test['releaseDate'] = test['releaseDate'].apply(lambda x: time.mktime(datetime.datetime.strptime(x, "%Y-%m-%d").timetuple()))

print 'Instantiating xgboost model...'
clf = XGBRegressor(  learning_rate=          0.2,
                     n_estimators=           100,
                     max_depth=              5,
                     subsample=              1.0,
                     colsample_bytree=       1.0,
                     objective=              'reg:linear',
                     nthread=                3,
                     seed=                   seed)

#december
print 'Training xgboost model...'
print gpfinal.dtypes

gpfinal['stock'] = gpfinal.units_x
test['stock'] = s
gpfinal.drop('units_x', axis=1, inplace=True)
gpfinal.drop('units_y', axis=1, inplace=True)


clf.fit(gpfinal, final_target, eval_metric='rmse',
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