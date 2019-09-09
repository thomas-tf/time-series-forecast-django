# time-series-forecast-django
Time Series forecast web app using statistically modelling, machine learning and deep learning algorithms.

# Live Demo
https://time-series-forecast-django.herokuapp.com/. Deep learning algorithms are disabled as resources are very limited using a free hosting plan.

# Deploy on Heroku
Simply clone this repo and connect to your cloned repo on Heroku, then click on Deploy branch.

# Run Locally
1. Get Python 3.6+

2. Run
```
pip install -r requirements.txt
python manage.py migrate
```

3. Then finally run
```
python manage.py runserver
```

4. Open your browser and go to http://127.0.0.1:8000

# Algorithms
Algorithms are stored in 
```
/tspredict/algorithms/
```

Feel free to add your own and alter 
```
/tspredict/views.py
/tspredict/templates/main/header.html
```
