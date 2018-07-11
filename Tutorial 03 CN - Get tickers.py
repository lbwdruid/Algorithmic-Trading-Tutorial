import tushare
import pandas
import datetime

# Step 1. Get tickers online
tickersRawData = tushare.get_stock_basics()
tickers = tickersRawData.index.tolist()

# Step 2. Save the ticker list to a local file
dateToday = datetime.datetime.today().strftime('%Y%m%d')
file = '../02. Data/00. TickerListCN/TickerList_'+dateToday+'.csv'
tickersRawData.to_csv(file)
print ('Tickers saved.')


