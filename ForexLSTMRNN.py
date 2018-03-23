
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Activation
from keras.callbacks import EarlyStopping
from keras.layers import LSTM
from keras.regularizers import L1L2
from math import sqrt
#import matplotlib
#be able to save images on server
#matplotlib.use('Agg')
from matplotlib import pyplot
import numpy as np

# date-time parsing function for loading the dataset
def parser(x):
    return datetime.strptime(x, '%d/%m/%Y')

# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]

# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]

# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'linear'))
    model.compile(loss='mean_squared_error',optimizer='adam')
    early_stopping = EarlyStopping(monitor = 'loss', patience=10)
    for i in range(nb_epoch):
        model.fit(X, y, epochs=1, batch_size=batch_size, callbacks=[early_stopping], verbose=0, shuffle=False)
        model.reset_states()
    return model

# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]


# load dataset
series = read_csv('USDVolumesData.csv', parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)


# summarize first few rows
#print(series.head())
# line plot
#series.plot()
#pyplot.ylabel('Rate')
#pyplot.savefig('/Users/ronrickarnaiz/Documents/NM/FOREX/newlstm1.png', bbox_inches = 'tight')
#pyplot.show()
#series.head()


# transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values, 1)
supervised_values = diff_values.values


# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values


# split data into train and test-sets
train, test = supervised_values[0:-12], supervised_values[-12:]


# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

# fit the model
lstm_model = fit_lstm(train_scaled, 1, 1000, 6)
# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(lstm_model, 1, X)
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    # invert differencing
    yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
    # store forecast
    predictions.append(yhat)
    expected = raw_values[len(train) + i + 1]
    print('day=%d, Predicted=%f, Expected=%f' % (i+1, yhat, expected))


# report performance
rmse = sqrt(mean_squared_error(raw_values[-8:], predictions))
mape = np.mean(np.abs(raw_values[-8:] - predictions) / raw_values[-8:]) * 100
print('Test RMSE: %.3f' % rmse)
print('Test MAPE: %.3f' % mape)
# line plot of observed vs predicted
#pyplot.plot(raw_values[-12:])
#pyplot.plot(predictions)
#pyplot.ylabel('TRADING_RATE')
#pyplot.xlabel('Day')
#pyplot.savefig('newlstm3.png',bbox_inches = 'tight')
#pyplot.show()



