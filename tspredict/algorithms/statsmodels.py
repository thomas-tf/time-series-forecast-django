import quandl
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima_model import ARMA
from datetime import timedelta
import statsmodels.api as sm
import math
from .utils import plot_ohlc

def holtwinters(stockid):

	df = quandl.get(f'HKEX/{stockid}', api_key="DRXLGbo1-dWHV-R86jxH")

	df.sort_index(inplace=True)
	df = df[-(365-(2*52)):]
	df = df.resample('B').ffill().reindex(pd.date_range(df.index[0],df.index[-1],freq='B'))
	df.interpolate(method='time', inplace=True)

	divident = math.floor(df.iloc[:-20].shape[0]/5)

	cv_score = np.array([])
	temp = divident
	for i in range(0,5):
		hw = ExponentialSmoothing(df['Nominal Price'].iloc[:temp], trend='add', seasonal='add', seasonal_periods=20).fit()
		prediction = hw.forecast(steps=20)
		cv_score = np.append(cv_score, mean_absolute_error(df['Nominal Price'].iloc[temp:temp+20], prediction))
		temp+=divident

	hw = ExponentialSmoothing(df['Nominal Price'], trend='add', seasonal='add', seasonal_periods=20).fit()

	prediction = hw.forecast(steps=20)

	prediction_dates = pd.date_range(start=df.index[-1],periods=20, freq='B')

	div = plot_ohlc(stockid, df, prediction, prediction_dates)

	predictions = dict(zip(prediction_dates, prediction))

	return predictions, cv_score, div


def arima(stockid):

	df = quandl.get(f'HKEX/{stockid}', api_key="DRXLGbo1-dWHV-R86jxH")

	df.sort_index(inplace=True)
	df = df[-(365-(2*52)):]
	df = df.resample('B').ffill().reindex(pd.date_range(df.index[0],df.index[-1],freq='B'))
	df.interpolate(method='time', inplace=True)

	divident = math.floor(df.iloc[:-20].shape[0]/5)

	cv_score = np.array([])
	temp = divident
	for i in range(0,5):
		arma = ARMA(df['Nominal Price'].iloc[:temp], (1, 1, 1)).fit()
		prediction = arma.forecast(steps=20)[0]
		cv_score = np.append(cv_score, mean_absolute_error(df['Nominal Price'].iloc[temp:temp+20], prediction))
		temp+=divident

	arma = ARMA(df['Nominal Price'], (1, 1, 1)).fit()

	prediction = arma.forecast(steps=20)[0]

	prediction_dates = pd.date_range(start=df.index[-1],periods=20, freq='B')

	div = plot_ohlc(stockid, df, prediction, prediction_dates)

	predictions = dict(zip(prediction_dates, prediction))

	return predictions, cv_score, div