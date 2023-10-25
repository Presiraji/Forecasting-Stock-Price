
import time
import yfinance as yf
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from datetime import timedelta


st.set_page_config(layout="wide")
st.write("""
# Previous Earnings Price Actions - Stay Goated

Enter the stock symbol and select a date range to view the stock closing prices for previous earnings.
""")

# User input for the ticker symbol
tickerSymbol = st.text_input("Enter Stock Ticker:", "SPY").upper()
day = st.slider("Day", min_value=2, max_value=252, value=15)
st.write("Days After", day)

# User input for start and end dates
start_date = st.date_input("Start Date YYYY-MM_DD", pd.to_datetime("2023-10-09"), key='Q1')
end_date = start_date + timedelta(days=day)
if st.button('Reload Data', key='1'):
# Display user inputs
    st.write(f"Showing data for: {tickerSymbol}")
st.write(f"From: {start_date} to: {end_date}")
# get data on this ticker
ALPHA1 = yf.Ticker(tickerSymbol)
# get the historical prices for this ticker
ALPHA = ALPHA1.history(period='1d', start=start_date, end=end_date)
# Display closing price chart with adjustable size
st.write("## Closing Price")
fig1 = px.line(ALPHA, y='Close', title='Closing Price')
fig1.update_layout(autosize=True, height=400)
st.plotly_chart(fig1, use_container_width=False)



# User input for start and end dates
start_date = st.date_input("Start Date YYYY-MM_DD", pd.to_datetime("2023-10-09"), key='Q2')
end_date = start_date + timedelta(days=day)
if st.button('Reload Data', key='2'):
# Display user inputs
    st.write(f"Showing data for: {tickerSymbol}")
st.write(f"From: {start_date} to: {end_date}")
# get data on this ticker
ALPHA1 = yf.Ticker(tickerSymbol)
# get the historical prices for this ticker
ALPHA = ALPHA1.history(period='1d', start=start_date, end=end_date)
# Display closing price chart with adjustable size
st.write("## Closing Price")
fig1 = px.line(ALPHA, y='Close', title='Closing Price')
fig1.update_layout(autosize=True, height=400)
st.plotly_chart(fig1, use_container_width=False)



# User input for start and end dates
start_date = st.date_input("Start Date YYYY-MM_DD", pd.to_datetime("2023-10-09"), key='Q3')
end_date = start_date + timedelta(days=day)
if st.button('Reload Data', key='3'):
# Display user inputs
    st.write(f"Showing data for: {tickerSymbol}")
st.write(f"From: {start_date} to: {end_date}")
# get data on this ticker
ALPHA1 = yf.Ticker(tickerSymbol)
# get the historical prices for this ticker
ALPHA = ALPHA1.history(period='1d', start=start_date, end=end_date)
# Display closing price chart with adjustable size
st.write("## Closing Price")
fig1 = px.line(ALPHA, y='Close', title='Closing Price')
fig1.update_layout(autosize=True, height=400)
st.plotly_chart(fig1, use_container_width=False)



# User input for start and end dates
start_date = st.date_input("Start Date YYYY-MM_DD", pd.to_datetime("2023-10-09"), key='Q4')
end_date = start_date + timedelta(days=day)
if st.button('Reload Data', key='4'):
# Display user inputs
    st.write(f"Showing data for: {tickerSymbol}")
st.write(f"From: {start_date} to: {end_date}")
# get data on this ticker
ALPHA1 = yf.Ticker(tickerSymbol)
# get the historical prices for this ticker
ALPHA = ALPHA1.history(period='1d', start=start_date, end=end_date)
# Display closing price chart with adjustable size
st.write("## Closing Price")
fig1 = px.line(ALPHA, y='Close', title='Closing Price')
fig1.update_layout(autosize=True, height=400)
st.plotly_chart(fig1, use_container_width=False)
