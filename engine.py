# ML Engine to predict stock price and trend
# Potential Algorithms - LSTM
# Have to analyze time series datasets
# Need to find place to get old data of SPY stock to train (Yahoo)
# Author: Trevor Zobrist

# Current TODO:
# Implement API?
# Hyperparameter tuning of model

# BEST PERFORMANCE METRICS TO DATE:
# -----SPY-----
# RMSE: 1.0505
# MAPE: 0.2360

# -----AAPL-----
# RMSE: 0.3006
# MAPE: 0.1568

# -----TSLA-----
# RMSE: 3.16683
# MAPE: 0.89600

# import packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import yfinance as yf
from datetime import date, timedelta
from time import sleep
import SMS

# import ML-specific packages
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout


# function to calculate Exponential Moving Average (EMA)
def calculate_ema(close_prices):
    ema = close_prices.ewm(com=0.2).mean()
    return ema


# regression metrics (RMSE, MAPE)
def calculate_rmse(y_true, y_pred):
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    return rmse


def calculate_mape(y_true, y_pred):
    y_pred, y_true = np.array(y_pred), np.array(y_true)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape


def forecast(current_data):
    forecast_values = [actual_stock_value[-1]]
    for i in range(10):
        X_future = np.array(current_data)
        X_future = np.reshape(X_future, (1, 60, 1))
        future_stock_value = regressor.predict(X_future, verbose=0)
        current_data = np.append(current_data, future_stock_value)
        current_data = current_data[1:]
        future_stock_value = scaler.inverse_transform(future_stock_value)
        forecast_values = np.append(forecast_values, future_stock_value)

    return forecast_values


# Historical Data from Yahoo (20y = 2002-2022)
# get current day and next
today = date.today()
tomorrow = date.today() + timedelta(1)

# get stock ticker from user
TICK = -1
while TICK != 'SPY' and TICK != 'AAPL' and TICK != 'TSLA':
    TICK = input("ENTER STOCK TICKER (SPY, AAPL, TSLA): ")
    if TICK == 'SPY' or TICK == 'AAPL' or TICK == 'TSLA':
        # download 20 years of data from yahoo, make index a column and rename to Date
        dataset = yf.download(tickers=TICK, period='20y', interval='1d')
        dataset.reset_index(inplace=True)
        dataset = dataset.rename(columns={'index': 'Date'})
    else:
        print("Invalid Input. Retry.")

# How much data to use to train?
# no input is max - 1 month (use month to test accuracy)
TRAIN = input("Amount of Training from Historical Data [max = " + str(len(dataset) - 1) + ", default = " + str(
    len(dataset) - 31) + "]: ")
if TRAIN == '':
    TRAIN = len(dataset) - 31
TRAIN = int(TRAIN)

# make datasets from closing prices
dataset_train = dataset['Close'].iloc[:TRAIN]
dataset_test = dataset['Close'].iloc[TRAIN:]

# full dataset and full dataset ema
dataset_total = dataset['Close']
dataset_total_ema = calculate_ema(dataset_total).values
dataset_total_ema = dataset_total_ema.reshape(-1, 1)

# separate closing price entries from training set and compute ema
training_set = dataset_train.values
training_set = training_set.reshape(-1, 1)

# separate closing stock prices from test dataset for comparison with prediction
actual_stock_value = dataset_test.values
actual_stock_value = actual_stock_value.reshape(-1, 1)

# calculate ema from test dataset entries for comparison with prediction
actual_stock_ema = calculate_ema(dataset_test).values
actual_stock_ema = actual_stock_ema.reshape(-1, 1)

# fit scaler (minmax)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler = scaler.fit(training_set)

# normalize training data
scaled_training_set = scaler.transform(training_set)

# create data structures for x_train/y_train
X_train, y_train = [], []
for i in range(60, TRAIN):
    X_train.append(scaled_training_set[i - 60:i, 0])
    y_train.append(scaled_training_set[i, 0])
X_train = np.array(X_train)
y_train = np.array(y_train)
# reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# gather input for sms messaging
sms = input("Send SMS Message? ")

# add option to use model without training or re-train
print("1 - Use current model, 2 - Re-train model")
train_int = -1
while train_int != 1 and train_int != 2:
    train_int = int(input("INPUT: "))
    if train_int == 2:
        # -------------------------ENGINE CODE-------------------------------------
        regressor = Sequential()
        regressor.add(LSTM(units=50,
                           return_sequences=True,
                           input_shape=(X_train.shape[1], 1)))
        # regressor.add(Dropout(0.2))
        regressor.add(LSTM(units=50,
                           return_sequences=True))
        # regressor.add(Dropout(0.2))
        regressor.add(LSTM(units=50))
        # regressor.add(Dropout(0.2))
        regressor.add(Dense(units=1))

        # fit model
        regressor.compile(optimizer='adam',
                          loss='mean_squared_error')
        history = regressor.fit(X_train, y_train, epochs=70, batch_size=32)

        # plot loss of model throughout training
        plt.plot(history.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.show()

        # save model weights
        if TICK == 'SPY':
            regressor.save('spy_model.h5')
        if TICK == 'TSLA':
            regressor.save('tsla_model.h5')
        if TICK == 'AAPL':
            regressor.save('aapl_model.h5')
        #############################################################################
    elif train_int == 1:
        # if not retraining just load model from saved weight files
        if TICK == 'SPY':
            regressor = tf.keras.models.load_model('spy_model.h5')
        if TICK == 'TSLA':
            regressor = tf.keras.models.load_model('tsla_model.h5')
        if TICK == 'AAPL':
            regressor = tf.keras.models.load_model('aapl_model.h5')

    else:
        print("Invalid Input. Retry.")

# model input
m_input = dataset_total[len(dataset_total) - len(dataset_test) - 59:].values
m_input = m_input.reshape(-1, 1)

# normalize model input
m_input = scaler.transform(m_input)

# create testing set data structures
X_test = []
for i in range(60, ((len(dataset) + 60) - TRAIN)):
    X_test.append(m_input[i - 60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# predict values from test set
predicted_stock_value = regressor.predict(X_test)
predicted_stock_value = scaler.inverse_transform(predicted_stock_value)

# forecast future stock value for next 10 days
current_dataset = yf.download(tickers=TICK, period='60d', interval='1d')
current_dataset.reset_index(inplace=True)
current_dataset = current_dataset.rename(columns={'index': 'Date'})
current_data = scaler.transform(current_dataset['Close'].values.reshape(-1, 1))
forecast = forecast(current_data)

# print performance metrics
rmse = calculate_rmse(actual_stock_value, predicted_stock_value)
print("RMSE: " + str(rmse))
mape = calculate_mape(actual_stock_value, predicted_stock_value)
print("MAPE: " + str(mape))

# send SMS message to phone
if sms == 'y' or sms == 'Y':
    SMS.send(TICK + " STOCK FORECAST\nModel RMSE: " + str(round(rmse, 4)) + "\nModel MAPE: " + str(round(mape, 4))
            + "\nCurrent Close: " + str(round(forecast[0], 3)))
    sleep(0.5) # delay so messages send in correct order
    SMS.send("--10 Day Forecast--" +
             "\nDay 1: " + str(round(forecast[1], 3)) +
             "\nDay 2: " + str(round(forecast[2], 3)) +
             "\nDay 3: " + str(round(forecast[3], 3)) +
             "\nDay 4: " + str(round(forecast[4], 3)) +
             "\nDay 5: " + str(round(forecast[5], 3)) +
             "\nDay 6: " + str(round(forecast[6], 3)) +
             "\nDay 7: " + str(round(forecast[7], 3)))
    sleep(0.5)
    SMS.send("\nDay 8: " + str(round(forecast[8], 3)) +
             "\nDay 9: " + str(round(forecast[9], 3)) +
             "\nDay 10: " + str(round(forecast[10], 3)))
    print("SMS sent")

# plot predicted and actual values
plt.plot(range(31),
         actual_stock_value,
         color='blue',
         label='Actual ' + TICK + ' Price')
plt.plot(range(31),
         predicted_stock_value,
         color='red',
         label='Predicted ' + TICK + ' Price',
         linestyle='--')
plt.plot(range(30, 41),
         forecast,
         color='green',
         label='Forecasted ' + TICK + ' Price',
         linestyle='--')
plt.title(TICK + ' Price Prediction Engine')
plt.xlabel('Current - Forecast')
plt.ylabel(TICK + ' Stock Price')
plt.xticks([])
plt.legend()
plt.savefig(TICK + '_engine_performance.jpg')
plt.show()
