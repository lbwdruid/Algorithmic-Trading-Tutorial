from ib.opt import Connection, message
from ib.ext.Contract import Contract
from ib.ext.Order import Order


ibConnection = None

def operate(orderId, ticker, action, quantity, price=None):
	# 1. Contract
	contract = Contract()
	contract.m_symbol = ticker
	contract.m_secType = 'STK'
	contract.m_exchange = 'ISLAND'
	contract.m_currency = 'USD'

	# 2. Order
	order = Order()
	if price is not None:
		order.m_orderType = 'LMT'
		order.m_lmtPrice = price
	else:
		order.m_orderType = 'MKT'
	order.m_totalQuantity = quantity
	order.m_action = action

	# 3. Place Order
	ibConnection.placeOrder(orderId, contract, order)

# Step 1. Establish connetion
ibConnection = Connection.create(port=7497, clientId=999)
ibConnection.connect()

# Step 2. Buy 123 NVDA
#operate(orderId=10, ticker='TSLA', action='BUY', quantity=123)
operate(orderId=11, ticker='TSLA', action='SELL', quantity=100)

# Step 3. Disconnect
ibConnection.disconnect()

