#  Stock Price Trend Prediction with LSTM

##  Objective
Predict future stock prices using historical trends and deep learning (LSTM).

##  Tools Used
- Python, Pandas, NumPy
- TensorFlow/Keras (LSTM)
- yfinance (data fetching)
- Matplotlib (visualization)
- Scikit-learn (normalization)
- Streamlit *(optional)*

##  Project Steps
1. **Data Collection**: Downloaded historical stock data using `yfinance`.
2. **Preprocessing**: Scaled closing prices & created time-series sequences.
3. **Modeling**: Built and trained a 2-layer LSTM model for price prediction.
4. **Evaluation**: Plotted actual vs predicted prices using matplotlib.
5. **Indicators**: Integrated 50-day Moving Average (MA) and Relative Strength Index (RSI).
6. **Model Saving**: Saved trained model as `lstm_model.h5`.
7. **Optional Dashboard**: Built a Streamlit app to visualize stock prices interactively.

##  Outputs
- Line plot of Actual vs Predicted prices
- MA50 and RSI indicator plots
- Trained model (`lstm_model.h5`)
- Streamlit dashboard (`stock_app.py`)

