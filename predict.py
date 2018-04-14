import pandas as pd
import numpy as np

train = pd.read_csv('train.csv', sep='|')
items = pd.read_csv('items.csv', sep='|')
prices = pd.read_csv('prices.csv', sep='|')



items_mask = items['releaseDate'].apply(lambda x: pd.Timestamp(x))


i_next_oct = items.where(items_mask > pd.Timestamp('2017-10-01')).dropna()
i_bet_oct_nov = items.where((items_mask > pd.Timestamp('2017-10-01')) & (items_mask < pd.Timestamp('2017-10-08'))).dropna()

items_price = pd.merge(i_bet_oct_nov, prices, on=['pid', 'size'], how='left')

ar = np.array([[]])
cont = 0
for d in np.arange('2017-10-01', '2018-02-28', dtype='datetime64[D]'):
	print str(cont)
	cont2 = 0
	cont += 1
	for pr in items_price[str(d)]:
		ar[str(d)][cont2] = pr
		cont2 += 1


i_before_oct = items.where(items_mask == pd.Timestamp('2017-10-01')).dropna()




#print (i_bet_oct_nov)
#merge = pd.merge(items, train, how=)