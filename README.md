# LSTM_Stock_Forecasting
Multistep time series forecasting of stock closing prices using Keras

Author: Trevor Zobrist

## main.py
Yahoo Finance API realtime chart display

## engine.py
Keras LSTM model that output 1 timestep predicition given last 60 timesteps

Model is trained on last 20 years of historical data from Yahoo API (or less if stock isn't as old)

After each training model's weight configuration is saved as .h5 file

One weight configuration per stock

## SMS.py
Uses SMTP and dedicated gmail email address to send SMS message to my phone (or any phone number as long as carrier is known) with selected stock's current closing price, next 10 closing prices and model's current accuracy metrics (Root Mean Squared Error and Mean Absolute Percentage Error)

Even if SMS message is not selected prediction is still modeled using matplotlib and output to computer
