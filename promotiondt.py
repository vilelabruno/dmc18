#%%
import pandas as pd
import numpy as np

train = pd.read_csv('train.csv', sep='|')
items = pd.read_csv('items.csv', sep='|')
prices2 = pd.read_csv('prices.csv', sep='|')
prices = pd.read_csv('prices.csv', sep='|')
aux = pd.DataFrame()
aux['promotionPrice'] = 0
prices['pid'] = prices['pid'].astype(str)
prices['size'] = prices['size'].astype(str)
items['pid'] = items['pid'].astype(str)
items['size'] = items['size'].astype(str)
prices['pid'] = prices[['pid','size']].apply('_'.join, axis=1)
items['pid'] = items[['pid','size']].apply('_'.join, axis=1)
prices.drop('size', axis=1, inplace=True)
items.drop('size', axis=1, inplace=True)
pricesInvert = prices.T

mg = pd.merge(items, prices, on='pid', how='right')
print mg
for col in pricesInvert.columns:
	initValue = pricesInvert[col][0]
	print pricesInvert[col][pricesInvert[col] < initValue]


#print (i_bet_oct_nov)
#merge = pd.merge(items, train, how=)