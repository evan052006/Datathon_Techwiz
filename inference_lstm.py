import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Grab the most recent stock data
data = yf.download('AAPL', period='40d')

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(data[['Close']])

# Grabbing the last 30 days of data for prediction
windows_size = 30
last_30_days = scaled_features[-windows_size:]

# Shaping the data to match the model input
x = np.reshape(last_30_days, (1, windows_size, 1))

# Loading model
model = load_model('Models/Stockly-lstm.keras')

# Predicting the next day's stock price
predictions = model.predict(x)
predictions = scaler.inverse_transform(predictions)

print(f"Predicted: ${predictions[0][0]:.2f}")
