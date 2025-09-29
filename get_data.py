import os
import pandas as pd
import datetime as dt

def get_data(symbols, dates, path="data"):
    """ Read stock data (adjusted close) for given symbols from CSV files in a 'data' directory."""

    # Initialize an empty DataFrame to hold the stock prices for all symbols
    df_final = pd.DataFrame(index=dates)

    # Loop through each symbol to read and process data
    for symbol in symbols:
        file_path = os.path.join(path, f"{symbol}.csv")

        # Read the CSV file for the given symbol
        df_temp = pd.read_csv(file_path,
                              index_col='Date',
                              parse_dates=True,
                              usecols=['Date', 'Adj Close'],
                              na_values='NaN')

        # Rename the 'Adj Close' column to the stock symbol
        df_temp = df_temp.rename(columns={'Adj Close': symbol})

        # Join the stock data with the main DataFrame (df_final)
        df_final = df_final.join(df_temp, how='left')

    return df_final


# Example usage inside your main script
if __name__ == "__main__":

    start_date = dt.datetime(2019, 1, 1)
    end_date = dt.datetime(2019, 12, 31)

    dates = pd.date_range(start_date, end_date)

    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM', 'SPY']  # Include SPY as a benchmark and test multiple symbols

    allocations = [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000
    risk_free_rate = 0.0
    sample_freq = 252
    # Get the stock data for the given symbols
    df_prices = get_data(symbols, dates)
    # Print first few rows to verify
    print(df_prices.head())

