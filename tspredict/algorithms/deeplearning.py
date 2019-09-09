import quandl
import pandas as pd
from numpy import array
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Flatten, TimeDistributed, Conv1D, MaxPooling1D
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from datetime import timedelta
from .utils import plot_ohlc

def lstm(stockid):
	# split a univariate sequence into samples
	def split_sequence(sequence, n_steps_in, n_steps_out):
		X, y = list(), list()
		for i in range(len(sequence)):
			# find the end of this pattern
			end_ix = i + n_steps_in
			out_end_ix = end_ix + n_steps_out
			# check if we are beyond the sequence
			if out_end_ix > len(sequence):
				break
			# gather input and output parts of the pattern
			seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
			X.append(seq_x)
			y.append(seq_y)
		return array(X), array(y)

	# define input sequence
	df = quandl.get(f'HKEX/{stockid}', api_key="DRXLGbo1-dWHV-R86jxH")

	df.sort_index(inplace=True)
	df = df[-(365-(2*52)):]

	df.drop(['Previous Close', 'Change (%)', 'Net Change', 'Lot Size', 'P/E(x)'], axis=1, inplace=True)

	df = df.interpolate(method='time')

	prediction_dates = pd.date_range(start=df.index[-1],periods=20, freq='B')

	raw_seq = df['Nominal Price'].values

	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaler.fit(raw_seq.reshape(-1, 1))
	raw_seq = scaler.transform(raw_seq.reshape(-1, 1)).reshape(1, -1)[0]

	# choose a number of time steps
	n_steps_in, n_steps_out = 100, 20
	# split into samples
	X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
	# reshape from [samples, timesteps] into [samples, timesteps, features]
	n_features = 1
	X = X.reshape(X.shape[0], X.shape[1], n_features)

	# define model
	def baseline_model():
		# create model
		model = Sequential()
		model.add(LSTM(50, activation='relu', input_shape=(n_steps_in, n_features)))
		model.add(Dense(n_steps_out))
		# Compile model
		model.compile(loss='mae', optimizer='adam')
		return model

	# fit model
	estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, verbose=0)
	tscv = TimeSeriesSplit(n_splits=5)
	cv_score = cross_val_score(estimator, X, y, scoring='neg_mean_absolute_error', cv=tscv)

	model = baseline_model()
	model.fit(X, y, epochs=100, verbose=0)
	# demonstrate prediction
	x_input = raw_seq[-n_steps_in:]
	x_input = x_input.reshape(1, n_steps_in, n_features)
	yhat = model.predict(x_input, verbose=0)
	prediction = scaler.inverse_transform(yhat.reshape(-1, 1)).reshape(1, -1)[0]

	div = plot_ohlc(stockid, df, prediction, prediction_dates)

	predictions = dict(zip(prediction_dates, prediction))

	return predictions, cv_score, div