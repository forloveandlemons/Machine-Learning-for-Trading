import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
from collections import OrderedDict
from util import get_data, plot_data
from portfolio.analysis import get_portfolio_value, get_portfolio_stats, plot_normalized_data
import LinRegLearner as lrl
import KNNLearner as knn
#import BagLearner as bl

def compute_portvals(start_date, end_date, orders_file, start_val):
	mydf=pd.read_csv(orders_file)
	mydf.sort(["Date"])
	mydf=mydf[mydf["Date"]>=start_date]
	mydf=mydf[mydf["Date"]<=end_date]
	
	dates=pd.date_range(start_date,end_date)
	
	#generate prices dataframe
	df_prices=pd.DataFrame()
	
	symbolSeries=mydf["Symbol"]
	symbolSeries=symbolSeries.drop_duplicates()
	symbolList=[]
	for each in symbolSeries:
		symbolList.append(each)
	df_prices=get_data(symbolList, dates, addSPY=True)
	df_prices.drop("SPY", axis=1, inplace=True)
	df_prices["CASH"]=1
	
	#generate trades dataframe
	df_trades=df_prices.copy()
	df_temp_positions=df_trades.copy()
	
	
	for each in df_trades:
		df_trades[each]=0
	for index in range(0,len(mydf)):
		trade_date=mydf["Date"].iloc[index]
		symbol=mydf["Symbol"].iloc[index]
		#order=1 if buy order=-1 if sell
		order=2*((mydf.loc[index,'Order']=="BUY")-0.5)
		shares=+mydf["Shares"].iloc[index]
		df_temp_positions=df_trades.copy()
		df_temp_positions.loc[trade_date,"CASH"]=start_val
		df_temp_positions=df_temp_positions.cumsum()
		#calculating leverage before each trade is entered
		long_position=0
		abs_short_position=0
		cash_position=start_val
		trade_value=shares*df_prices.loc[trade_date,symbol]
		leverage=1.0
		
		for each in df_temp_positions:
			if (each!="CASH" and df_temp_positions.loc[trade_date,each]>0):
				long_position+=trade_value
				cash_position-=trade_value
			if (each!="CASH" and df_temp_positions.loc[trade_date,each]<0):
				abs_short_position+=trade_value
				cash_position+=trade_value
			leverage=(long_position+abs_short_position)/(long_position-abs_short_position+cash_position)
		if leverage<=2:
			df_trades.loc[trade_date,symbol]+=order*shares
			df_trades.loc[trade_date,"CASH"]+=-order*shares*df_prices.loc[trade_date,symbol]
			
	#generate position dataframe
	df_positions=df_trades.copy()
	df_positions.loc[start_date,"CASH"]=start_val+df_positions.loc[start_date,"CASH"]
	df_positions=df_positions.cumsum()
	#generate portfolio value dataframe
	df_values=df_prices.mul(df_positions)
	df_values=df_values.sum(axis=1)
	portvals=df_values
	return portvals
	
def simple_moving_average(start_date, end_date, df_values, window):
	#calculate sma
	df_time_values=df_values.copy()
	df_sma=pd.rolling_mean(df_time_values,window)
	return df_sma

def upper_band(start_date, end_date, df_values, window):
	#calculate upperband
	df_time_values=df_values.copy()
	df_sma=pd.rolling_mean(df_time_values,window)
	df_rolling_std=pd.rolling_std(df_time_values,window)
	return df_rolling_std*2+df_sma
	
def lower_band(start_date, end_date, df_values, window):
	#calculate lowerband
	df_time_values=df_values.copy()
	df_sma=pd.rolling_mean(df_time_values,window)
	df_rolling_std=pd.rolling_std(df_time_values,window)
	return -df_rolling_std*2+df_sma
	
def bb_value(start_date, end_date, df_values, window):
	df_price_values=df_values.copy()
	df_sma_values=pd.rolling_mean(df_price_values,window)
	df_rolling_std=pd.rolling_std(df_price_values,window)
	df_bbvalues=(df_price_values-df_sma_values)/(2*df_rolling_std)
	return df_bbvalues.dropna()
	
def momentum_value(start_date, end_date, df_values, window):
	df_current=df_values.copy()
	df_ndaysback=df_values.shift(window)
	df_momentum=df_current/df_ndaysback-1
	return df_momentum.dropna()
	
def volatility_value(start_date, end_date, df_values, window):
	df_volatility=pd.rolling_std(df_values, window)
	df_vol=(df_volatility-df_volatility.mean())/(2*df_volatility.std())
	return df_vol.dropna()
	
def return_value(start_date, end_date, df_values, window):
	df_5daysshiftback=df_values.shift(-5)
	df_current=df_values
	df_return=df_5daysshiftback/df_current-1
	df_return=df_return.dropna()
	return df_return
	
def data_generator(df_x1, df_x2, df_x3, df_y, training_file):
	df_all=pd.merge(pd.DataFrame(df_x1), pd.DataFrame(df_x2), left_index=True, right_index=True)
	df_all=pd.merge(pd.DataFrame(df_all), pd.DataFrame(df_x3), left_index=True, right_index=True)
	df_all=pd.merge(pd.DataFrame(df_all), pd.DataFrame(df_y), left_index=True, right_index=True)
	df_all.columns=['X1','X2','X3','Y']
	df_all=df_all.dropna()
	df_all.to_csv(training_file, index=False, header=False)
	return df_all

    
def train_data(training_file):
	#training and query
    inf = open(training_file)
    data = np.array([map(float,s.strip().split(',')) for s in inf.readlines()])

    # separate out training and testing data
    trainX = data[:,0:-1]
    trainY = data[:,-1]

    # create a learner and train it
    learner = knn.KNNLearner(4) # create a LinRegLearner
    #learner=lrl.LinRegLearner()
    learner.addEvidence(trainX, trainY) # train it

    # evaluate in sample
    predY = learner.query(trainX) # get the predictions
    rmse = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])
    print
    print "In sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=trainY)
    print "corr: ", c[0,1]
    return learner, predY
    
def test_data(learner, test_file):
    
    inf_test = open(test_file)
    data = np.array([map(float,s.strip().split(',')) for s in inf_test.readlines()])
    testX = data[:,0:-1]
    testY = data[:,-1]
    # evaluate out of sample
    predY = learner.query(testX) # get the predictions
    rmse = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])
    print
    print "Out of sample results"
    print "RMSE: ", rmse
    c = np.corrcoef(predY, y=testY)
    print "corr: ", c[0,1]
    return predY
    


def all_trades(df_pred):
	#the function for implementing all exit and enter long&short trades
	df_temp=df_pred.copy()
	size=len(df_pred.index)
	long_entry=[]
	long_exit=[]
	short_entry=[]
	short_exit=[]
	df_trade=pd.DataFrame(0, index=df_pred.index, columns=['trade'])
	df_position=pd.DataFrame(0, index=df_pred.index, columns=['position'])
	
	for i in range(0, size-5):
		df_position=df_trade.cumsum()
				
		if df_position.iloc[i-1].item()==0:
			if df_pred.ix[i].item()>0.015:
				df_trade.iloc[i]=+100
				df_trade.iloc[i+5]=-100
				long_entry.append(df_temp.index[i])
				long_exit.append(df_temp.index[i+5])
				
			if df_pred.ix[i].item()<-0.015:
				df_trade.iloc[i]=-100
				df_trade.iloc[i+5]=+100
				short_entry.append(df_temp.index[i])
				short_exit.append(df_temp.index[i+5])

				
	df_position=df_trade.cumsum()
	return df_trade, df_position, long_entry, long_exit, short_entry, short_exit
	
def order_file(df_trade, orders_file):
	#generate the orderfile
	df_trade=df_trade[df_trade['trade']!=0]
	size=len(df_trade)
	index=range(0,size)
	columns=['Date', 'Symbol', 'Order', 'Shares']
	df_trade_file=pd.DataFrame(0, index=index, columns=columns)
	df_trade_file['Symbol']='IBM'
	df_trade_file['Shares']=100
	for i in range(0, size):
		df_trade_file.loc[i,'Date']=df_trade.index[i].date()
		if df_trade.iloc[i].item()==100:
			df_trade_file.loc[i,'Order']='BUY'
		if df_trade.iloc[i].item()==-100:
			df_trade_file.loc[i,'Order']='SELL'
	df_trade_file.to_csv(orders_file, index=False)
				


def test_run():
    """Driver function."""
    
    
    
    #////////////////////////////////////////////////////////////////////////////
    # Define input parameters
    print '******training***********'
    start_date = '2007-12-31'
    end_date = '2009-12-31'
    symbol_allocations = OrderedDict([('AAPL', 0.0), ('HPQ', 0.0), ('IBM', 1.0), ('HNZ', 0.0)])  # allocations from wiki example
    symbols = symbol_allocations.keys()  # list of symbols, e.g.: ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocs = symbol_allocations.values()  # list of allocations, e.g.: [0.2, 0.3, 0.4, 0.1]
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(symbols, dates)  # automatically adds SPY
    prices = prices_all[symbols]  # only portfolio symbols
    start_val=1*prices['IBM'][0]
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later
    
	# Get daily portfolio value
    portvals = get_portfolio_value(prices, allocs, start_val=prices['IBM'][0])
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # if a DataFrame is returned select the first column to get a Series
    
    
    #x1,x2,x3
    df_x1=bb_value(start_date, end_date, portvals, 12)
    df_x2=momentum_value(start_date, end_date, portvals, 7)
    df_x3=volatility_value(start_date, end_date, portvals, 10)
    df_y=return_value(start_date, end_date, portvals, 5)
    df_all=data_generator(df_x1, df_x2,df_x3,df_y, 'training.csv')
    df_index=df_all.index
    
    
    df_portvals=pd.DataFrame(portvals)
    learner, pred_Y_list=train_data('training.csv')
    df_predy=pd.DataFrame(pred_Y_list, index=df_index, columns=['pred_y'])
    df_y_predy=pd.merge(pd.DataFrame(df_y), pd.DataFrame(df_predy), left_index=True, right_index=True)
    df_y_predy.columns=['y','pred_y']
    df_y_adj=(1+df_y)*portvals
    df_predy_adj=(1+df_predy)*portvals
    df_y_predy_adjusted=pd.merge(pd.DataFrame(df_y_adj), pd.DataFrame(df_predy_adj), left_index=True, right_index=True)
    df_y_predy_adjusted.columns=['y_adj','pred_y_adj']
    df_y_predy_adjusted.dropna(how='all')
    df_y_three=pd.merge(pd.DataFrame(df_y_predy_adjusted), pd.DataFrame(df_portvals), left_index=True, right_index=True)
    df_y_three.columns=['Training Y','Predicted Y','Price']
    #plt.figure()
    #generate zooming in plot for Training Y/Price/Predicted Y
    ax=df_y_three.plot(xlim=['2008-01-31','2009-01-31'])
    plt.show()
    
    
    #trade based on predicted values
    df_trade, df_position, long_entry, long_exit, short_entry, short_exit=all_trades(df_predy)
    #generate order file
    orders_file='trainingorders.csv'
    order_file(df_trade, orders_file)
    #plot illustrating entry and exits
    ax = portvals.plot(fontsize=10, label='IBM',legend=True)
    ax.set_xlim(['2008-01-31','2009-01-31'])
    ax.set_xlabel("Date")
    ax.set_ylabel("price")
    
    
    #plot the vertical entry and exit signals
    for each in long_entry:
    	ax.axvline(each,color='lightgreen')	
    for each in long_exit:
    	ax.axvline(each,color='black')
    for each in short_entry:
    	ax.axvline(each,color='r')
    for each in short_exit:
    	ax.axvline(each,color='black')
    
    plt.show()
 	
 	
    #CHART OF BACKTEST
    start_port_val=10000
    #compute portfolio values
    strategy_portvals=compute_portvals(start_date, end_date, orders_file, start_port_val)
    # Get portfolio stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(strategy_portvals)
	# Simulate a SPY-only reference portfolio to get stats
    prices_SPX = get_data(['SPY'], pd.date_range(start_date, end_date))
    prices_SPX = prices_SPX[['SPY']]  # remove SPY
    portvals_SPX = get_portfolio_value(prices_SPX, [1.0])
    cum_ret_SPX, avg_daily_ret_SPX, std_daily_ret_SPX, sharpe_ratio_SPX = get_portfolio_stats(portvals_SPX)

    # Compare portfolio against SPY
    print "Data Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY: {}".format(sharpe_ratio_SPX)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY: {}".format(cum_ret_SPX)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY: {}".format(std_daily_ret_SPX)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY: {}".format(avg_daily_ret_SPX)
    print
    print "Final Portfolio Value: {}".format(strategy_portvals[-1])
    
    #plot the normalized backtest chart with benchmark being spy
    df_backtest_chart = pd.concat([strategy_portvals, prices_SPX['SPY']],  keys=['Portfolio', 'SPY'], axis=1)
    plot_normalized_data(df_backtest_chart, title="Daily portfolio value and SPY")
    
    
    #////////////////////////////////////////////////////////////////////////////
    #TEST THE 2010 DATA
    print '******testing***********'
    start_date = '2009-12-31'
    end_date = '2010-12-31'
    symbol_allocations = OrderedDict([('IBM', 0.0), ('HPQ', 0.0), ('IBM', 1.0), ('HNZ', 0.0)])  # allocations from wiki example
    symbols = symbol_allocations.keys()  # list of symbols, e.g.: ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocs = symbol_allocations.values()  # list of allocations, e.g.: [0.2, 0.3, 0.4, 0.1]
    dates = pd.date_range(start_date, end_date)
    prices_all = get_data(symbols, dates)  # automatically adds SPY
    prices = prices_all[symbols]  # only portfolio symbols
    start_val=1*prices['IBM'][0]
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later
    
	# Get daily portfolio value
    portvals = get_portfolio_value(prices, allocs, start_val=prices['IBM'][0])
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # if a DataFrame is returned select the first column to get a Series
    
    
    #x1,x2,x3
    df_x1=bb_value(start_date, end_date, portvals, 12)
    df_x2=momentum_value(start_date, end_date, portvals, 7)
    df_x3=volatility_value(start_date, end_date, portvals, 10)
    df_y=return_value(start_date, end_date, portvals, 5)
    #print df_y
    df_all=data_generator(df_x1, df_x2,df_x3,df_y, 'testing.csv')
    #print df_all
    df_index=df_all.index
    #print df_all
    
    
    
    
    
    pred_Y_list=test_data(learner,'testing.csv')
    df_portvals=pd.DataFrame(portvals)
    df_predy=pd.DataFrame(pred_Y_list, index=df_index, columns=['pred_y'])
    #print df_predy
    df_y_predy=pd.merge(pd.DataFrame(df_y), pd.DataFrame(df_predy), left_index=True, right_index=True)
    df_y_predy.columns=['y','pred_y']
    
    
    #trade based on predicted values
    df_trade, df_position, long_entry, long_exit, short_entry, short_exit=all_trades(df_predy)
    #generate order file
    orders_file='testingorders.csv'
    order_file(df_trade, orders_file)
    #plot illustrating entry and exits
    ax = portvals.plot(fontsize=10, label='IBM',legend=True)
    ax.set_xlabel("Date")
    ax.set_ylabel("price")
    
    #plot the vertical entry and exit signals
    for each in long_entry:
    	ax.axvline(each,color='lightgreen')	
    for each in long_exit:
    	ax.axvline(each,color='black')
    for each in short_entry:
    	ax.axvline(each,color='r')
    for each in short_exit:
    	ax.axvline(each,color='black')
    
    plt.show()
 	
 	
    #CHART OF BACKTEST
    start_port_val=10000
    #compute portfolio values
    strategy_portvals=compute_portvals(start_date, end_date, orders_file, start_port_val)
    # Get portfolio stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(strategy_portvals)
	# Simulate a SPY-only reference portfolio to get stats
    prices_SPX = get_data(['SPY'], pd.date_range(start_date, end_date))
    prices_SPX = prices_SPX[['SPY']]  # remove SPY
    portvals_SPX = get_portfolio_value(prices_SPX, [1.0])
    cum_ret_SPX, avg_daily_ret_SPX, std_daily_ret_SPX, sharpe_ratio_SPX = get_portfolio_stats(portvals_SPX)

    # Compare portfolio against SPY
    print "Data Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of SPY: {}".format(sharpe_ratio_SPX)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of SPY: {}".format(cum_ret_SPX)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of SPY: {}".format(std_daily_ret_SPX)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of SPY: {}".format(avg_daily_ret_SPX)
    print
    print "Final Portfolio Value: {}".format(strategy_portvals[-1])
    
    #plot the normalized backtest chart with benchmark being spy
    df_backtest_chart = pd.concat([strategy_portvals, prices_SPX['SPY']],  keys=['Portfolio', 'SPY'], axis=1)
    plot_normalized_data(df_backtest_chart, title="Daily portfolio value and SPY")
   

if __name__ == "__main__":
    test_run()
