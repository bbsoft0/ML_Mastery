# Time Series Forecasting
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
from pandas import concat
from pandas import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt
from pandas.plotting import autocorrelation_plot
from pandas import Grouper
from pandas import DataFrame
from matplotlib import pyplot
from pandas import read_csv
series = read_csv('./inputMastery/daily-births.csv', header=0, index_col=0)

# Explore
print(series.head())
print(series.size)
print(series.describe())

# Lesson 03: Data Visualization
series = read_csv('./inputMastery/shampoo-sales.csv', header=0, index_col=0)
series.plot(style='k.')
pyplot.show()

series.hist()
pyplot.show()

series = read_csv('./inputMastery/daily-births.csv', header=0,
                  index_col=0, parse_dates=True, squeeze=True)
groups = series.groupby(Grouper(freq='A'))
years = DataFrame()
for name, group in groups:
    years[name.year] = group.values
years = years.T
pyplot.matshow(years, interpolation=None, aspect='auto')
pyplot.show()

series = read_csv('./inputMastery/shampoo-sales.csv', header=0,
                  index_col=0, parse_dates=True, squeeze=True)
autocorrelation_plot(series)
pyplot.show()

# Lesson 04: Persistence Forecast Model


def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')


series = read_csv('./inputMastery/shampoo-sales.csv', header=0,
                  parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
# Create lagged dataset
values = DataFrame(series.values)
dataframe = concat([values.shift(1), values], axis=1)
dataframe.columns = ['t-1', 't+1']
print(dataframe.head(5))

# split into train and test sets
X = dataframe.values
train_size = int(len(X) * 0.66)
train, test = X[1:train_size], X[train_size:]
train_X, train_y = train[:, 0], train[:, 1]
test_X, test_y = test[:, 0], test[:, 1]

# persistence model


def model_persistence(x):
    return x


# walk-forward validation
predictions = list()
for x in test_X:
    yhat = model_persistence(x)
    predictions.append(yhat)
test_score = mean_squared_error(test_y, predictions)
print('Test MSE: %.3f' % test_score)

# plot predictions and expected results
pyplot.plot(train_y)
pyplot.plot([None for i in train_y] + [x for x in test_y])
pyplot.plot([None for i in train_y] + [x for x in predictions])
pyplot.show()

# Lesson 05: Autoregressive Forecast Model
# create and evaluate an updated autoregressive model
# load dataset
series = read_csv('./inputMastery/daily-min-temperatures.csv', header=0,
                  index_col=0, parse_dates=True, squeeze=True)
# split dataset
X = series.values
train, test = X[1:len(X)-7], X[len(X)-7:]
# train autoregression
window = 29
model = AutoReg(train, lags=29)
model_fit = model.fit()
coef = model_fit.params
# walk forward over time steps in test
history = train[len(train)-window:]
history = [history[i] for i in range(len(history))]
predictions = list()
for t in range(len(test)):
    length = len(history)
    lag = [history[i] for i in range(length-window, length)]
    yhat = coef[0]
    for d in range(window):
        yhat += coef[d+1] * lag[window-d-1]
    obs = test[t]
    predictions.append(yhat)
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()

# Lesson 06: ARIMA Forecast Model
# fit an ARIMA model and plot residual errors
# load dataset


def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')


series = read_csv('./inputMastery/shampoo-sales.csv', header=0, index_col=0,
                  parse_dates=True, squeeze=True, date_parser=parser)
series.index = series.index.to_period('M')
# fit model
model = ARIMA(series, order=(5, 1, 0))
model_fit = model.fit()
# summary of fit model
print(model_fit.summary())
# line plot of residuals
residuals = DataFrame(model_fit.resid)
residuals.plot()
pyplot.show()
# density plot of residuals
residuals.plot(kind='kde')
pyplot.show()
# summary stats of residuals
print(residuals.describe())


###################################################
# evaluate an ARIMA model using a walk-forward validation
# load dataset
def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')


series = read_csv('./inputMastery/shampoo-sales.csv', header=0, index_col=0,
                  parse_dates=True, squeeze=True, date_parser=parser)
series.index = series.index.to_period('M')
# split into train and test sets
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
# walk-forward validation
for t in range(len(test)):
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
