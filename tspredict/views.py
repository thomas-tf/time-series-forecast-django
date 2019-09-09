from django.shortcuts import render, redirect
from django.http import HttpResponse, HttpResponseRedirect
from django.contrib import messages

from .algorithms import utils, regression, deeplearning, statsmodels


def index(request):
	if request.method=='POST':
		stockid = request.POST['stockid']
		algo = request.POST['algo']

		# Validate stock iD
		if len(stockid)<5:
			while len(stockid)<5:
				stockid = '0' + stockid

		if len(stockid)>5:
			message.error(request, "Stock ID cannot be more than 5 digits")
			return HttpResponseRedirect(self.request.path_info)

		validate = utils.check_if_stock_exists(stockid)

		if(validate=="True"):
			if(algo=="lr"):
				predictions, cv_scores, div = regression.lasso_regression(stockid)
				str_cv_score = {}
				for i, s in enumerate(cv_scores):
					str_cv_score[f'{i+1}'] = s*-1
				return render(request, 'main/results.html', {
						'predictions':predictions,
						'cv_scores':str_cv_score,
						'cv_mean':cv_scores.mean()*-1,
						'cv_std':cv_scores.std(),
						'div':div,
						'algo':'Linear Regression'
					})
			elif(algo=="lstm"):
				predictions, cv_scores, div = deeplearning.lstm(stockid)
				str_cv_score = {}
				for i, s in enumerate(cv_scores):
					str_cv_score[f'{i+1}'] = s*-1
				return render(request, 'main/results.html', {
						'predictions':predictions,
						'cv_scores':str_cv_score,
						'cv_mean':cv_scores.mean()*-1,
						'cv_std':cv_scores.std(),
						'div':div,
						'algo':'LSTM'
					})
			elif(algo=="hw"):
				predictions, cv_scores, div = statsmodels.holtwinters(stockid)
				str_cv_score = {}
				for i, s in enumerate(cv_scores):
					str_cv_score[f'{i+1}'] = s
				return render(request, 'main/results.html', {
						'predictions':predictions,
						'cv_scores':str_cv_score,
						'cv_mean':cv_scores.mean(),
						'cv_std':cv_scores.std(),
						'div':div,
						'algo':'Holt-winters'
					})
			elif(algo=="arima"):
				predictions, cv_scores, div = statsmodels.arima(stockid)
				str_cv_score = {}
				for i, s in enumerate(cv_scores):
					str_cv_score[f'{i+1}'] = s
				return render(request, 'main/results.html', {
						'predictions':predictions,
						'cv_scores':str_cv_score,
						'cv_mean':cv_scores.mean(),
						'cv_std':cv_scores.std(),
						'div':div,
						'algo':'ARIMA'
					})
			elif(algo=="svr"):
				predictions, cv_scores, div = regression.support_vector_regressor(stockid)
				str_cv_score = {}
				for i, s in enumerate(cv_scores):
					str_cv_score[f'{i+1}'] = s*-1
				return render(request, 'main/results.html', {
						'predictions':predictions,
						'cv_scores':str_cv_score,
						'cv_mean':cv_scores.mean()*-1,
						'cv_std':cv_scores.std(),
						'div':div,
						'algo':'Support Vector Regressor'
					})
			
		else:
			messages.error(request, validate)

	return render(request, 'main/index.html')

def forecast(request):
	return HttpResponse("Hello, world. You're at the polls index.")
