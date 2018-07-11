import tushare
import pandas
import datetime
import os

def stockPriceIntraday(ticker, folder):
	# Step 1. Get intraday data online
	intraday = tushare.get_hist_data(ticker, ktype='5')

	# Step 2. If the history exists, append
	file = folder+'/'+ticker+'.csv'
	if os.path.exists(file):
		history = pandas.read_csv(file, index_col=0)
		intraday.append(history)

	# Step 3. Inverse based on index
	intraday.sort_index(inplace=True)
	intraday.index.name = 'timestamp'

	# Step 4. Save
	intraday.to_csv(file)
	print ('Intraday for ['+ticker+'] got.')

# Step 1. Get tickers online
tickersRawData = tushare.get_stock_basics()
tickers = tickersRawData.index.tolist()

# Step 2. Save the ticker list to a local file
dateToday = datetime.datetime.today().strftime('%Y%m%d')
file = '../02. Data/00. TickerListCN/TickerList_'+dateToday+'.csv'
tickersRawData.to_csv(file)
print ('Tickers saved.')

# Step 3. Get stock price (intraday) for all
for i, ticker in enumerate(tickers):
	try:
		print ('Intraday', i, '/', len(tickers))
		stockPriceIntraday(ticker, folder='../02. Data/01. IntradayCN')
	except:
		pass
print ('Intraday for all stocks got.')

