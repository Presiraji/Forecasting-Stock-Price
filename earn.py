
import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import timedelta
import os
import tempfile
import base64
import plotly.express as px
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import plotly.io as pio

# Function to save a plotly figure as a PNG file
def save_plot(fig, filename):
    pio.write_image(fig, filename)

# Function to create a PDF from PNG files
def create_pdf(png_files, pdf_filename):
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    width, height = letter
    num_images = len(png_files)
    
    # Calculate the size and position for each image
    image_width = width - 150  # Set image width (leave some margin)
    image_height = (height - 12) / num_images  # Distribute images vertically
    y_position = height - 0 - image_height  # Start from the top
    
    for png_file in png_files:
        c.drawImage(png_file, 0, y_position, width=image_width, height=image_height)
        y_position -= image_height + 1  # Move to the next position (with some margin)
    
    c.showPage()
    c.save()

# Function to plot and save chart
def plot_and_save(ticker_symbol, start_date, end_date, file_prefix):
    ALPHA1 = yf.Ticker(ticker_symbol)
    ALPHA = ALPHA1.history(period='1d', start=start_date, end=end_date)
    fig = px.line(ALPHA, y='Close', title='Closing Price')
    
    # Add date annotation
    date_annotation = f"{ticker_symbol}: {start_date.strftime('%Y-%m-%d')} to: {end_date.strftime('%Y-%m-%d')}"
    fig.add_annotation(x=0.50, y=0.50, xref="paper", yref="paper",
                       text=date_annotation, showarrow=False, font=dict(size=13))
    
    fig.update_layout(autosize=True, height=400, margin=dict(t=40, b=40))
    st.plotly_chart(fig, use_container_width=False)
    
    filename = f"{file_prefix}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.png"
    save_plot(fig, filename)
    return filename

# Setting up Streamlit page configuration
st.set_page_config(layout="wide")
st.write("""
# Previous Earnings Price Actions - Stay Goated

Enter the stock symbol and select a date range to view the stock closing prices for previous earnings.
""")

# User input for the ticker symbol
ticker_symbol = st.text_input("Enter Stock Ticker:", "SPY").upper()
day = st.slider("Day", min_value=2, max_value=252, value=15)
st.write("Days After", day)

# Initialize a list to store the names of the saved PNG files
saved_plots = []

# Loop through the 4 quarters
for i in range(1, 5):
    key_suffix = f"_Q{i}"
    start_date = st.date_input(f"Start Date YYYY-MM_DD (Q{i})", pd.to_datetime("2023-10-09"), key='start_date' + key_suffix)
    end_date = start_date + timedelta(days=day)
    if st.button('Reload Data', key='reload_data' + key_suffix):
        st.write(f"Showing data for: {ticker_symbol}")
    st.write(f"From: {start_date} to: {end_date}")
    saved_plot_filename = plot_and_save(ticker_symbol, start_date, end_date, f"quarter_{i}")
    saved_plots.append(saved_plot_filename)

# Create a button to save all plots as a PDF
if st.button('Save as PDF'):
    with tempfile.TemporaryDirectory() as tmpdirname:
        pdf_filename = os.path.join(tmpdirname, 'ticker_symbol-plots.pdf')
        create_pdf(saved_plots, pdf_filename)
        with open(pdf_filename, "rb") as f:
            pdf_file = f.read()
        b64_pdf = base64.b64encode(pdf_file).decode('utf-8')
        pdf_link = f'<a href="data:application/octet-stream;base64,{b64_pdf}" download="{ticker_symbol}-saved_plots.pdf">Download PDF</a>'
        st.markdown(pdf_link, unsafe_allow_html=True)

    # Clean up: Delete the PNG files after creating the PDF
    for saved_plot in saved_plots:
        if os.path.exists(saved_plot):
            os.remove(saved_plot)
