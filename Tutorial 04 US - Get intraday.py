import requests
import pandas
import io
import datetime
import os
import time

def dataframeFromUrl(url):
	dataString = requests.get(url).content
	parsedResult = pandas.read_csv(io.StringIO(dataString.decode('utf-8')), index_col=0)
	return parsedResult

def stockPriceIntraday(ticker, folder):
	# Step 1. Get data online
	url = 'https://www.alphavantage.co/query?apikey=NGV2US07ZZQULVGP&function=TIME_SERIES_INTRADAY&symbol={ticker}&interval=1min&outputsize=full&datatype=csv'.format(ticker=ticker)
	intraday = dataframeFromUrl(url)

	# Step 2. Append if history exists
	file = folder+'/'+ticker+'.csv'
	if os.path.exists(file):
		history = pandas.read_csv(file, index_col=0)
		intraday.append(history)

	# Step 3. Inverse based on index
	intraday.sort_index(inplace=True)

	# Step 4. Save
	intraday.to_csv(file)
	print ('Intraday for ['+ticker+'] got.')

# Step 1. Get ticker list online
tickersRawData = dataframeFromUrl('http://www.nasdaq.com/screening/companies-by-industry.aspx?exchange=NASDAQ&render=download')
tickers = tickersRawData.index.tolist()

# Step 2. Save the ticker list to a local file
dateToday = datetime.datetime.today().strftime('%Y%m%d')
file = '../02. Data/00. TickerListUS/TickerList'+dateToday+'.csv'
tickersRawData.to_csv(file)
print ('Tickers saved.')

# Step 3. Get stock price (intraday)
for i, ticker in enumerate(tickers):
	try:
		print ('Intraday', i, '/', len(tickers))
		stockPriceIntraday(ticker, folder='../02. Data/01. IntradayUS')
		time.sleep(2)
	except:
		pass
print ('Intraday for all stocks got.')




