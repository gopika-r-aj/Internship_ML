import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt

st.title("📈 Stock Price Viewer")

ticker = st.text_input("Enter Stock Ticker", "AAPL")
data = yf.download(ticker, start="2020-01-01")

st.write("## Close Price Chart")
st.line_chart(data['Close'])

st.write("## Last 5 Days Data")
st.dataframe(data.tail())
