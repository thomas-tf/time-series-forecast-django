import quandl
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from datetime import timedelta
from .utils import plot_ohlc, feature_engineering_regression

def linear_regression(stockid):
	df = quandl.get(f'HKEX/{stockid}', api_key="DRXLGbo1-dWHV-R86jxH")

	df = feature_engineering_regression(df)

	data = df[df.notnull().all(axis=1)].copy()
	forecast = df[df.isnull().any(axis=1)].copy()

	scaler = StandardScaler()
	poly = PolynomialFeatures(degree=3)

	categoricals = ['is quarter start', 'future is quarter start', 'date']
	numerics = list(set(data.columns.tolist()) - set(categoricals) - set(['label']))

	polyed = poly.fit_transform(data[numerics])
	data.drop(numerics, axis=1, inplace=True)
	data = pd.merge(data, pd.DataFrame(scaler.fit_transform(polyed), index=data.index), how='outer', left_index=True, right_index=True)

	tscv = TimeSeriesSplit(n_splits=5)

	cv_score = cross_val_score(Lasso(alpha=0.1), data.drop('label',axis=1), y=data.label, scoring='neg_mean_absolute_error', cv=tscv)

	model = Lasso(alpha=0.1)
	model.fit(data.drop('label',axis=1), y=data.label)

	forecast.drop('label',axis=1, inplace=True)

	forecast_polyed = poly.transform(forecast[numerics])
	forecast.drop(numerics, axis=1, inplace=True)
	forecast = pd.merge(forecast, pd.DataFrame(scaler.transform(forecast_polyed), index=forecast.index), how='outer', left_index=True, right_index=True)

	prediction = model.predict(forecast)

	prediction_dates = pd.date_range(start=df.index[-1],periods=20, freq='B')

	div = plot_ohlc(stockid, df, prediction, prediction_dates)

	predictions = dict(zip(prediction_dates, prediction))

	return predictions, cv_score, div


def support_vector_regressor(stockid):

	df = quandl.get(f'HKEX/{stockid}', api_key="DRXLGbo1-dWHV-R86jxH")

	df = feature_engineering_regression(df)

	data = df[df.notnull().all(axis=1)].copy()
	forecast = df[df.isnull().any(axis=1)].copy()

	scaler = StandardScaler()
	poly = PolynomialFeatures(degree=3)

	categoricals = ['is quarter start', 'future is quarter start']
	numerics = list(set(data.columns.tolist()) - set(categoricals) - set(['label']))

	polyed = poly.fit_transform(data[numerics])
	data.drop(numerics, axis=1, inplace=True)
	data = pd.merge(data, pd.DataFrame(scaler.fit_transform(polyed), index=data.index), how='outer', left_index=True, right_index=True)

	tscv = TimeSeriesSplit(n_splits=5)

	cv_score = cross_val_score(SVR(), data.drop('label',axis=1), y=data.label, scoring='neg_mean_absolute_error', cv=tscv)

	model = SVR()
	model.fit(data.drop('label',axis=1), y=data.label)

	forecast.drop('label',axis=1, inplace=True)

	forecast_polyed = poly.transform(forecast[numerics])
	forecast.drop(numerics, axis=1, inplace=True)
	forecast = pd.merge(forecast, pd.DataFrame(scaler.transform(forecast_polyed), index=forecast.index), how='outer', left_index=True, right_index=True)

	prediction = model.predict(forecast)

	prediction_dates = pd.date_range(start=df.index[-1],periods=20, freq='B')

	div = plot_ohlc(stockid, df, prediction, prediction_dates)

	predictions = dict(zip(prediction_dates, prediction))

	return predictions, cv_score, div
