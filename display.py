# Stock Closing Price analysis
# What it needs to do:
#   Get real time stock info. (probably Yahoo finance?)
#   Analyze information with ML/AI (Keras)
#   Make and output predictions
# Author: Trevor Zobrist

# import relevant packages
import yfinance as yf
import plotly.graph_objs as go

# get stock data... thanks Yahoo :]
data = yf.download(tickers='SPY', period='20y', interval='1d')
data.to_csv('historical_spy.txt')

# declare figure
fig = go.Figure()

# configure candlesticks
fig.add_trace(go.Candlestick(x=data.index,
                             open=data['Open'],
                             high=data['High'],
                             low=data['Low'],
                             close=data['Close'],
                             name='market data'))

# add titles
fig.update_layout(title='SPY Stock Price',
                  yaxis_title='Stock Price (USD per share)')

# x axis and range selection
fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=10, label="10m", step="minute", stepmode="backward"),
            dict(count=30, label="30m", step="minute", stepmode="backward"),
            dict(count=1, label="1h", step="hour", stepmode="backward"),
            dict(count=6, label="6h", step="hour", stepmode="backward"),
            dict(step="all")
        ])
    )
)

# show figure
fig.show()
