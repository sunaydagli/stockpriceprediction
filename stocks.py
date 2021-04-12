# Keras, Tensorflow
# Packages: pandas, pandas_datareader

import pandas_datareader as pdr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import numpy as np
import tensorflow as tf
import math

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from numpy import array

# get the data from Tiingo
data = pdr.get_data_tiingo('AAPL', api_key='56f9823528c78f57d1a3c772b702d7d67086d844')
data.to_csv('APPL.csv')
data = pd.read_csv('APPL.csv')
data_close = data.reset_index()['close']
# plt.plot(data_close)
# plt.show()

# apply MinMax scalar because LSTM is sensitive to scale 
scaler = MinMaxScaler(feature_range=(0,1))
data_close = scaler.fit_transform(np.array(data_close).reshape(-1,1))

# splitting data into train and test split
training_size=int(len(data_close) * 0.65)
test_size = len(data_close) - training_size
train_data, test_data = data_close[0:training_size,:], data_close[training_size:len(data_close), :1]

# helper method to create np arrays
def create_dataset(dataset, time_step=1):
    dataX = []
    dataY = []
    for i in  range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

# reshape into X = t, t + 1, ... Y = t + stamp
time_step = 150
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# reshape intput to be suitable for LSTM: [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Create Stacked LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(100,1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# print(model.summary())

# Train model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, verbose=1)

# Prediction & performance metrics
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Transform back to original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Calculate RMSE performance metrics
mse_trained = math.sqrt(mean_squared_error(y_train, train_predict))
print('Mean Squared Error Trained: ' + str(mse_trained))

# Test Data RMSE
mse_test = math.sqrt(mean_squared_error(y_test, test_predict))
print('Mean Squared Error Test: ' + str(mse_test))

# Plotting

look_back = 100
trainPredictPlot = np.empty_like(data_close)
trainPredictPlot[:,:] = np.nan
trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(data_close)
testPredictPlot[:,:] = np.nan
testPredictPlot[len(train_predict) + (look_back * 2) + 1 : len(data_close) - 1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(data_close))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()

# total 441 entries, taking top 100
x_input = test_data[341:].reshape(1,-1)
temp_input = list(x_input)
temp_input = temp_input[0].tolist()

#demonstrate prediction for next 30 days
lst_output = []
n_steps = 100
i = 0
while (i < 30):
    if (len(temp_input) > 100):
        # print(temp_input)
        x_input = np.array(temp_input[1:])
        print("{} day input {}".format(i, x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i, yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
        i = i + 1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose = 0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i = i + 1
print(lst_output)

# Plotting 

day_new = np.arange(1,101)
day_pred = np.arange(101, 131)

data3 = data_close.tolist()
data3.extend(lst_output)
plt2.plot(day_new, scaler.inverse_transform(data_close[1158:]))
plt2.plot(day_pred, scaler.inverse_transform(lst_output))
# plt2.plot(data3[1200:])
plt2.show()