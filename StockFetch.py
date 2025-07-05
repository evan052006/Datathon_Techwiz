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
        data_year1.dropna(inplace=True)
        data_year5.dropna(inplace=True)

        # Write each datasets onto a .csv file
        data_year1.to_csv(f'Datasets/{name}_Year1.csv', index=False)
        data_year5.to_csv(f'Datasets/{name}_Year5.csv', index=False)
