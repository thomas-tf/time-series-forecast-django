import quandl
import plotly.offline as opy
import plotly.graph_objs as go
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import timedelta

def check_if_stock_exists(stockid):
	if stockid.isdigit():
		try:
			df = quandl.get(f'HKEX/{stockid}', api_key="DRXLGbo1-dWHV-R86jxH", rows=1)
		except Exception as e:
			return e

		if df.empty:
			return "Data is empty"

		return "True"


def fetch_data(stockid):
	df = quandl.get(f'HKEX/{stockid}', api_key="DRXLGbo1-dWHV-R86jxH", rows=30*6)
	return df

def plot_ohlc(stockid, df, prediction, prediction_dates):
	fig = go.Figure(data=[
		go.Candlestick(
			name=f'Stock: {stockid}',
			x=df.index,
			open=df['Ask'],
			high=df['High'],
			low=df['Low'],
			close=df['Nominal Price'],
			showlegend=False
		),
		go.Scatter(
			name='Forecast', 
			y=prediction, 
			x=prediction_dates, 
			line=dict(color='royalblue', width=2)
		)], 
		layout_title_text=f'Hong Kong Stock Exchange {stockid}',
		layout=go.Layout(
			template='ggplot2',
			paper_bgcolor='rgba(38, 50, 56, 100)',
			plot_bgcolor='rgba(38, 50, 56, 100)',
			font=dict(family='Roboto, Sans-serif', size=18, color='#ededed'),
			showlegend=True,
			xaxis=dict(
				showgrid=False, 
				zeroline=False,
			),
			yaxis=dict(
				showgrid=True,
				gridcolor='#ededed',
			)
		)
	)


	div = opy.plot(fig, auto_open=False, output_type='div')

	return div

def feature_engineering_regression(df):

	df.sort_index(inplace=True)
	df = df[-(365-(2*52)):]

	df.drop(['Previous Close', 'Change (%)', 'Net Change', 'Lot Size', 'P/E(x)'], axis=1, inplace=True)

	df = df.interpolate(method='time')

	df['weekday'] = df.index.weekday
	df['quarter'] = df.index.quarter
	df['is quarter start'] = df.index.is_quarter_start
	df['is quarter start'] = df['is quarter start'].map({True:1, False:0})
	df['future date'] = df.index + timedelta(7)
	df['future weekday'] = df['future date'].dt.weekday
	df['future quarter'] = df['future date'].dt.quarter
	df['future is quarter start'] = df['future date'].dt.is_quarter_start
	df['future is quarter start'] = df['future is quarter start'].map({True:1, False:0})

	le = LabelEncoder()
	df['date'] = le.fit_transform(df.index)

	df.drop('future date', axis=1, inplace=True)

	df['label'] = df['Nominal Price'].shift(-20)

	return df