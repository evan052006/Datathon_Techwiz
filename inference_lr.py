from joblib import load
import yfinance as yf

# Grab the most recent stock data
data = yf.download('AAPL', period='40d')

# Adding SMA (Simple Moving Average) into the data
data['SMA_5'] = data['Close'].rolling(window=5).mean()
data['SMA_10'] = data['Close'].rolling(window=10).mean()

# Adding EMA (Exponential Moving Average) into the data
data['EMA_5'] = data['Close'].ewm(span=5, adjust=False).mean()
data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()

# Adding RSI (Relative Strength Index) into the data
delta = data['Close'].diff()
avg_gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
avg_loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# Grab the features of the final stock price to predict the next day stock
feature = data[['Open', 'High', 'Low', 'Close', 'Volume', 'SMA_5', 'SMA_10', 'EMA_5', 'EMA_10', 'RSI']].tail().values

# Load the models in your directory
model = load('Models/Stockly-lr.joblib')
predictions = model.predict(feature)

# Predicts gives the next day stock
print(f"Predicted: ${predictions[0]:.2f}")

