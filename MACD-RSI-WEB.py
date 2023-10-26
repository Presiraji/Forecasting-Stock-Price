
import time
import yfinance as yf
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(layout="wide")

# Function to hide Streamlit branding and sidebar
def hide_streamlit_branding():
    st.markdown("""
        <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

# Call the function to apply the style
hide_streamlit_branding()

st.write("""
# Forecasting Future Value - Stay Goated

Enter the stock symbol and select a date range to view the stock closing prices and volume.
""")

# User input for the ticker symbol
tickerSymbol = st.text_input("Enter Stock Ticker:", "SPY").upper()

import streamlit as st

# Slider for the S
st.write('Seasonal Period (s): This is the length of the seasonal cycle. For monthly data with an annual pattern, s would be 12. For quarterly data, s might be 4. But for Daily we use 22 as in 22 trading days in one month')

# Adding a slider
S = st.slider('Choose a value for S:', min_value=12, max_value=252, value=22)

# Display the chosen value
st.write('The current value of S is:', S)


# User input for start and end dates
start_date = st.date_input("Start Date YYYY-MM_DD", pd.to_datetime("2021-10-09"))
end_date = st.date_input("End Date YYYY-MM_DD ", pd.to_datetime("2023-10-09"))


if st.button('Reload Data'):

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
fig1.update_layout(autosize=True, height=700)
st.plotly_chart(fig1, use_container_width=True)

# Display volume chart with adjustable size
st.write("## Volume Price")
fig2 = px.line(ALPHA, y='Volume', title='Volume Price')
fig2.update_layout(autosize=True, height=700)
st.plotly_chart(fig2, use_container_width=True)


a = ALPHA


# Calculate MACD and Signal Line
a['12_EMA'] = a['Close'].ewm(span=12, adjust=False).mean()
a['26_EMA'] = a['Close'].ewm(span=26, adjust=False).mean()
a['MACD'] = a['12_EMA'] - a['26_EMA']
a['Signal_Line'] = a['MACD'].ewm(span=9, adjust=False).mean()

# Calculate RSI
delta = a['Close'].diff()
gain = (delta.where(delta > 0, 0)).fillna(0)
loss = (-delta.where(delta < 0, 0)).fillna(0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
a['RSI'] = 100 - (100 / (1 + rs))

# Calculate Bollinger Bands
a['20_SMA'] = a['Close'].rolling(window=20).mean()
a['Std_Dev'] = a['Close'].rolling(window=20).std()
a['Bollinger_Lower'] = a['20_SMA'] - (2 * a['Std_Dev'])
a['Bollinger_Upper'] = a['20_SMA'] + (2 * a['Std_Dev'])

# Plot MACD
plt.figure(figsize=(22,6), dpi=100)
plt.title('MACD & Signal Line')
a['MACD'][-200:].plot(label='MACD', color='blue', legend=True)
a['Signal_Line'][-200:].plot(label='Signal Line', color='red', linestyle='--', legend=True)
# Rotate x-axis labels for better visibility and declutter the x axis
plt.show()



# Plot RSI
plt.figure(figsize=(22,6), dpi=100)
plt.title('Relative Strength Index (RSI)')
a['RSI'][-200:].plot(label='RSI', color='purple', legend=True)
plt.axhline(70, color='red', linestyle='--')
plt.axhline(30, color='green', linestyle='--')
plt.fill_between(a.index[-200:], 70, 30, color='yellow', alpha=0.1)
# Rotate x-axis labels for better visibility and declutter the x axis
plt.show()

# Plot Bollinger Bands
plt.figure(figsize=(22,6), dpi=100)
plt.title('Bollinger Bands')
a['Close'][-200:].plot(label='Close Price', color='blue', legend=True)
a['Bollinger_Lower'][-200:].plot(label='Lower Band', color='red', linestyle='--', legend=True)
a['Bollinger_Upper'][-200:].plot(label='Upper Band', color='green', linestyle='--', legend=True)
plt.fill_between(a.index[-200:], a['Bollinger_Lower'][-200:], a['Bollinger_Upper'][-200:], color='yellow', alpha=0.1)
# Rotate x-axis labels for better visibility and declutter the x axis
plt.show()







# MACD Plot
st.write("## MACD & Signal Line")
fig_macd = go.Figure()
fig_macd.add_trace(go.Scatter(x=a.index[-200:], y=a['MACD'][-200:], mode='lines', name='MACD'))
fig_macd.add_trace(go.Scatter(x=a.index[-200:], y=a['Signal_Line'][-200:], mode='lines', name='Signal Line', line=dict(dash='dash')))
fig_macd.update_layout(title='MACD & Signal Line', autosize=True, height=700)
st.plotly_chart(fig_macd, use_container_width=True)

# RSI Plot
st.write("## Relative Strength Index (RSI)")
fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=a.index[-200:], y=a['RSI'][-200:], mode='lines', name='RSI'))
fig_rsi.add_shape(type='line', line=dict(dash='dash', color='red'), y0=70, y1=70, x0=a.index[-200], x1=a.index[-1])
fig_rsi.add_shape(type='line', line=dict(dash='dash', color='green'), y0=30, y1=30, x0=a.index[-200], x1=a.index[-1])
fig_rsi.update_layout(title='Relative Strength Index (RSI)', autosize=True, height=700)
st.plotly_chart(fig_rsi, use_container_width=True)

# Bollinger Bands Plot
st.write("## Bollinger Bands")
fig_bb = go.Figure()
fig_bb.add_trace(go.Scatter(x=a.index[-200:], y=a['Close'][-200:], mode='lines', name='Close Price'))
fig_bb.add_trace(go.Scatter(x=a.index[-200:], y=a['Bollinger_Lower'][-200:], mode='lines', name='Lower Band', line=dict(dash='dash')))
fig_bb.add_trace(go.Scatter(x=a.index[-200:], y=a['Bollinger_Upper'][-200:], mode='lines', name='Upper Band', line=dict(dash='dash')))
fig_bb.update_layout(title='Bollinger Bands', autosize=True, height=700)
st.plotly_chart(fig_bb, use_container_width=True)










































####################################################################################################################################################








import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from pmdarima import auto_arima
import datetime
import plotly.express as px
import plotly.graph_objects as go
import time

if st.button('Run SARIMAX Model'):
    st.info("Model is running, please wait...")
    
    # Adding a progress bar
    progress_bar = st.progress(0)
    
    # Assuming ALPHA is a pre-existing DataFrame. Creating a copy of it in 'df'.
    df = ALPHA.copy()
    progress_bar.progress(5)

    df['Date'] = pd.to_datetime(df.index) #--------------
    df2 = df.set_index('Date') #---------------
    
    # Check if 'Close' column exists, if not replace 'Close' with the correct column name
    if 'Close' not in df.columns:
        st.error("Error: 'Close' column not found in the DataFrame. Please replace 'Close' with the correct column name.")
        st.stop()
    progress_bar.progress(10)

    # Setting the 'Date' column as the new index of the DataFrame 'df'.
    df.set_index('Date', inplace=True)
    progress_bar.progress(20)

    # Extracting values from the 'Close' column.
    data = df["Close"].values
    progress_bar.progress(25)

    # Defining the training and testing data split percentage.
    split_percentage = 0.80
    split_index = int(len(data) * split_percentage)

    # Splitting the data.
    x_train, x_test = data[:split_index], data[split_index:]
    progress_bar.progress(30)

    # Displaying the lengths of the training and testing datasets.
    st.write("Training data length:", len(x_train), "Testing data length:", len(x_test))
    progress_bar.progress(35)

    # Finding the best ARIMA model parameters.
    stepwise_fit = auto_arima(data, trace=True, suppress_warnings=True)
    stepwise_fit.summary()
    progress_bar.progress(50)

    # Extracting the ARIMA order.
    arima_order = stepwise_fit.order
    p, d, q = arima_order
    progress_bar.progress(55)

    # Defining seasonal period and SARIMAX model.
    model = sm.tsa.statespace.SARIMAX(data, order=(p, d, q), seasonal_order=(p, d, q, S))
    model_fit = model.fit()
    progress_bar.progress(70)

    # Assuming you have already fitted your model and it's named 'model_fit'
    start = len(x_train)
    end = len(x_train) + len(x_test) - 1
    pred = model_fit.predict(start=start, end=end)

    progress_bar.progress(80)

    # Calculating error metrics.
    rmse = np.sqrt(mean_squared_error(x_test, pred))
    r2 = r2_score(x_test, pred)

    # Displaying error metrics.
    st.write("Root Mean Squared Error:", rmse)
    st.write("R-squared:", r2)
    progress_bar.progress(85)
    
    
    #Predict the Future
    pred_future = model_fit.predict(start=end, end=end+15)
    length = len(pred_future)
    progress_bar.progress(87)
    
   # Importing the datetime module to work with dates.
    import datetime
    progress_bar.progress(88)

    # Importing the pandas library for data manipulation.
    import pandas as pd

    # Getting the second to last date from the 'Date' column of the 'ALPHA' dataframe.
    last_date = ALPHA.index[-1]
    progress_bar.progress(89)

    # Calculating the next business day after 'last_date'.
    start_date = last_date + pd.tseries.offsets.BDay(1)
    progress_bar.progress(90)

    # Generating a list of 23 consecutive dates starting from 'start_date'.
    dates = [start_date + datetime.timedelta(days=idx) for idx in range(30)]  # Increased range to ensure enough market dates
    progress_bar.progress(91)

    # Filtering out the weekend dates from the 'dates' list.
    market_dates = [date for date in dates if date.weekday() < 5][:len(pred_future)]  # Use only the first len(pred_future) market dates

    progress_bar.progress(92)

    # Creating a pandas series with 'pred_future' values and 'market_dates' as index.
    pred_future2 = pd.Series(pred_future,market_dates)
    progress_bar.progress(93)

    # Resetting index of 'pred_future2' series to get a dataframe with 'Date' and 'Prediction' columns.
    dfuture = pred_future2.reset_index()
    progress_bar.progress(94)

    # Renaming the columns of 'df' to 'Date' and 'Prediction'.
    dfuture.columns = ['Date', 'Prediction']

    df = pd.DataFrame(dfuture)

    # Renaming the columns
    df.columns = ['Date', 'Prediction']

    # Streamlit app
    st.title('Forecasted Future Values')
    st.header('15 Days')
    st.write(df)
    progress_bar.progress(95)


    # Ensure you've imported and defined df, df2, and pred_future2 before this code
    progress_bar.progress(96)

    # Display date vs prediction chart
    st.write("## Date vs Prediction")
    fig2 = px.line(df, x='Date', y='Prediction', title='Date vs Prediction', markers=True)
    fig2.update_traces(line=dict(color='blue'))
    fig2.update_layout(grid=dict(rows=1, columns=1), plot_bgcolor='rgba(0,0,0,0)')
    fig2.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey')
    fig2.update_layout(autosize=True, height=700)
    st.plotly_chart(fig2, use_container_width=True)
    progress_bar.progress(98)

    
    # Importing the required libraries
    import pandas as pd
    import streamlit as st
    import plotly.express as px

    # Constructing a Pandas Series for the predicted future stock prices, with a datetime index.
    # pred_future: A list or array-like structure holding the predicted future stock prices.
    # market_dates: A list or array-like structure holding the corresponding dates for the predictions.
    pred_future2 = pd.Series(pred_future, index=pd.to_datetime(market_dates))

    # Using Streamlit to display a title for the stock price chart.
    st.write("## Stock Price")

    # Selecting the last 756 rows of the DataFrame df2 to focus on more recent data.
    # This assumes that df2 is a Pandas DataFrame holding stock price information.
    df2_sub = df2[-756:]

    # Converting the index of df2_sub to datetime format to ensure proper alignment and plotting.
    df2_sub.index = pd.to_datetime(df2_sub.index)

    # Creating a line chart of the closing stock prices using Plotly Express.
    # x=df2_sub.index: Setting the x-axis to the datetime index of df2_sub.
    # y='Close': Setting the y-axis to the 'Close' column of df2_sub, representing closing stock prices.
    fig1 = px.line(df2_sub, x=df2_sub.index, y='Close', title='Stock Price')

    # Adding a line to the chart for the predicted future stock prices.
    # x=pred_future2.index: Setting the x-axis to the datetime index of the predicted prices.
    # y=pred_future2: Setting the y-axis to the predicted future stock prices.
    # mode='lines': Specifying that the predicted prices should be plotted as a line.
    # name='Future Predicted Price': Setting the legend name for the predicted prices line.
    fig1.add_scatter(x=pred_future2.index, y=pred_future2, mode='lines', name='Future Predicted Price')

    # Setting the layout of the chart, with a specified width and height.
    # autosize=False: Disabling autosize to manually set the size of the chart.
    # width=1200 and height=700: Setting the width and height of the chart.
    fig2.update_layout(autosize=True, height=700)

    # Using Streamlit to display the chart on the web application.
    # use_container_width=False: Specifying that the chart should not adjust its width to the container.
    st.plotly_chart(fig1, use_container_width=True)







    # Update progress bar
    progress_bar.progress(99)

    
    progress_bar.progress(100)
    st.success("Model run successfully!")

