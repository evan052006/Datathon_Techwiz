import yfinance as yf
import pandas as pd

if __name__ == '__main__':
    tickers = {'AppleStock': 'AAPL',
              'MicrosoftStock': 'MSFT',
              'GoogleStock': 'GOOGL',
              'BCAStock': 'BBCA.JK',
              'BRIStock': 'BBRI.JK',
              'MandiriStock': 'BMRI.JK',
              'McDonaldsStock': 'MCD',
              'StarbucksStock': 'SBUX',
              'TargetStock': 'TGT'}

    for name, ticker in tickers.items():
        # Gather stock datasets using yfinance
        data_year1 = yf.download(ticker, start='2024-01-01', end='2025-01-01')
        data_year5 = yf.download(ticker, start='2020-01-01', end='2025-01-01')
        
        # Add target column for prediction
        data_year1['Target'] = data_year1['Close'].shift(-1)
        data_year5['Target'] = data_year5['Close'].shift(-1)
        data_year1.reset_index(inplace=True)
        data_year5.reset_index(inplace=True)

        # Adding SMA (Simple Moving Average)
        data_year1['SMA_5'] = data_year1['Close'].rolling(window=5).mean()
        data_year1['SMA_10'] = data_year1['Close'].rolling(window=10).mean()
        data_year5['SMA_5'] = data_year5['Close'].rolling(window=5).mean()
        data_year5['SMA_10'] = data_year5['Close'].rolling(window=10).mean()

        # Adding EMA (Exponential Moving Average)
        data_year1['EMA_5'] = data_year1['Close'].ewm(span=5, adjust=False).mean()
        data_year1['EMA_10'] = data_year1['Close'].ewm(span=10, adjust=False).mean()
        data_year5['EMA_5'] = data_year5['Close'].ewm(span=5, adjust=False).mean()
        data_year5['EMA_10'] = data_year5['Close'].ewm(span=10, adjust=False).mean()

        # Adding RSI (Relative Strength Index)
        delta = data_year1['Close'].diff()
        avg_gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        avg_loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data_year1['RSI'] = 100 - (100 / (1 + rs))

        delta = data_year5['Close'].diff()
        avg_gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        avg_loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data_year5['RSI'] = 100 - (100 / (1 + rs))

        # Clears any missing data
        data_year1.dropna(inplace=True)
        data_year5.dropna(inplace=True)

        # Write each datasets onto a .csv file
        data_year1.to_csv(f'Datasets/{name}_Year1.csv', index=False)
        data_year5.to_csv(f'Datasets/{name}_Year5.csv', index=False)
