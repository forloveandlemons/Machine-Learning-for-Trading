"""MC2-P1: Market simulator."""

import pandas as pd
import numpy as np
import os

from util import get_data, plot_data
from portfolio.analysis import get_portfolio_value, get_portfolio_stats, plot_normalized_data

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
		order=2*((mydf["Order"].iloc[index]=="BUY")-0.5)
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


def test_run():
    """Driver function."""
    # Define input parameters
    start_date = '2011-01-14'
    end_date = '2011-12-14'
    orders_file = os.path.join("orders", "orders2.csv")
    start_val = 1000000

    # Process orders
    portvals = compute_portvals(start_date, end_date, orders_file, start_val)
    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]]  # if a DataFrame is returned select the first column to get a Series
    
    # Get portfolio stats
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio = get_portfolio_stats(portvals)

    # Simulate a $SPX-only reference portfolio to get stats
    prices_SPX = get_data(['$SPX'], pd.date_range(start_date, end_date))
    prices_SPX = prices_SPX[['$SPX']]  # remove SPY
    portvals_SPX = get_portfolio_value(prices_SPX, [1.0])
    cum_ret_SPX, avg_daily_ret_SPX, std_daily_ret_SPX, sharpe_ratio_SPX = get_portfolio_stats(portvals_SPX)

    # Compare portfolio against $SPX
    print "Data Range: {} to {}".format(start_date, end_date)
    print
    print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    print "Sharpe Ratio of $SPX: {}".format(sharpe_ratio_SPX)
    print
    print "Cumulative Return of Fund: {}".format(cum_ret)
    print "Cumulative Return of $SPX: {}".format(cum_ret_SPX)
    print
    print "Standard Deviation of Fund: {}".format(std_daily_ret)
    print "Standard Deviation of $SPX: {}".format(std_daily_ret_SPX)
    print
    print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    print "Average Daily Return of $SPX: {}".format(avg_daily_ret_SPX)
    print
    print "Final Portfolio Value: {}".format(portvals[-1])

    # Plot computed daily portfolio value
    df_temp = pd.concat([portvals, prices_SPX['$SPX']], keys=['Portfolio', '$SPX'], axis=1)
    plot_normalized_data(df_temp, title="Daily portfolio value and $SPX")


if __name__ == "__main__":
    test_run()
