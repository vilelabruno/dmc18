from __future__ import division
print 'Loading libraries ...'

import numpy as np
from xgboost.sklearn import XGBRegressor
import pandas as pd
import os
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

seed = 1234
np.random.seed(seed)
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
prices = pd.read_csv('newprices.csv')
ss = pd.read_csv('sampleSubmission.csv')

gp = train.groupby('pid').agg({'date': np.max})
gp = pd.merge(gp, train, how='left')
count_color = train_it.groupby('group')['unit'].count().reset_index()
count_group.columns = ['group', 'count_group']
train_it = pd.merge(train_it, count_group, on='group', how='left', sort=False)
train_it.drop('group', axis=1, inplace=True)





print 'Done!'