from statistics import correlation

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

from get_data import get_data

def plot_normalized_data(df, title="Normalized prices", xlabel="Date", ylabel="Normalized price"):
    """Normalize given stock prices and plot for comparison."""
    # TODO: Implement normalization and plotting
    # Plot comparison with market (SPY)
    print("TODO: Implement normalization and plotting")
    return

def get_portfolio_stats(port_val, market_val, daily_rf=0.0, samples_per_year=252.0, calc_optional=False):
    """Calculate portfolio statistics: core and optional metrics."""

    # TODO: Calculate core statistics (cumulative return, avg daily return, volatility, Sharpe ratio)
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio, beta, alpha = 0, 0, 0, 0, 0, 0

    optional_stats = {}
    if calc_optional:
        # TODO: Implement optional statistics
        print("**TODO: Implement optional statistics")

        optional_stats = {
            'Sortino Ratio': 0,
            'Tracking Error': 0,
            'Information Ratio': 0,
        }

    return cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio, beta, alpha, optional_stats


def get_portfolio_value(prices, allocs, start_val=1000000):
    """Compute daily portfolio value given stock prices, allocations, and starting value."""
    # Step 1.1: Normalize prices by dividing by first day prices
    normed = prices / prices.iloc[0]
    
    # Step 1.2: Calculate daily portfolio value
    # Multiply normalized prices by allocations and starting capital, then sum across stocks
    port_val = (normed * allocs * start_val).sum(axis=1)
    
    return port_val

def assess_portfolio(sd, ed, syms, allocs, sv=1000000, rfr=0.0, sf=252.0, gen_plot=True, calc_optional=False):
    """Assess the portfolio and calculate relevant statistics."""
    dates = pd.date_range(sd, ed)

    # TODO: Get stock data and SPY data
    prices = get_data(syms, dates)
    prices_SPY = get_data(['SPY'], dates)
    print("TODO: Get stock data and SPY data")

    # TODO: Get portfolio value
    port_val = get_portfolio_value(prices, allocs, sv)
    print("TODO: Get portfolio value")

    # TODO: Get portfolio statistics - call function
    ev = port_val  # Placeholder for end value
    cr, adr, sddr, sr, beta, alpha, optional_stats = 0, 0, 0, 0, 0, 0, {}
    print("TODO: Get portfolio statistics")

    # TODO: Compute correlation with the market (SPY)
    print("TODO: Compute correlation with the market (SPY)")
    # Hint: First, compute the daily returns for both the portfolio and SPY.
    # Then, use the .corr() function in pandas to calculate the correlation
    # between these two series.
    correlation = 0  # Placeholder for end value

    # TODO: Implement plotting if `gen_plot` is True
    print("TODO: Implement plotting if `gen_plot` is Trues")
    plot_normalized_data(prices, title="Portfolio vs Market (SPY)")

    print_portfolio_stats(cr, adr, sddr, sr, sv, ev, correlation, beta, alpha, optional_stats)
    return cr, adr, sddr, sr, ev, beta, alpha, optional_stats


def print_portfolio_stats(cr, adr, sddr, sr, sv, ev, correlation, beta, alpha, optional_stats):
    """Pretty print portfolio statistics with optional statistics."""
    # TODO: Implement Pretty Printing - e.g., see HW description
    print("TODO: Implement Pretty Printing - e.g., see HW description")
    return # TODO: remove return - make printing prety

    print(f"Cumulative Return: {cr}")
    print(f"Avg Daily Return: {adr}")
    print(f"Volatility (Std Dev): {sddr}")
    print(f"Sharpe Ratio: {sr}")
    print(f"{'Correlation with SPY:':<30} {correlation:>12,.4f}")  # Use commas for thousands and 2 decimal places for money
    print(f"Beta: {beta}")
    print(f"Alpha: {alpha}")
    print(f"Start Portfolio Value: {sv}")
    print(f"End Portfolio Value: {ev}")

    if optional_stats:
        print("\nAdditional Statistics:")
        for stat, value in optional_stats.items():
            print(f"{stat}: {value}")


# Testing the function with a specific set of stocks and allocations
if __name__ == "__main__":
    start_date = dt.datetime(2019, 1, 2)
    end_date = dt.datetime(2019, 1, 8)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations = [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000
    risk_free_rate = 0.0
    sample_freq = 252

    # Get data for testing
    dates = pd.date_range(start_date, end_date)
    prices = get_data(symbols, dates)
    
    # Step 1.1: Normalize prices
    normed = prices / prices.iloc[0]
    print('"normed" type =', type(normed))
    print(normed)
    print()
    
    # Step 1.2: Calculate Daily Portfolio Value
    print("Step 1.2: Calculate Daily Portfolio Value")
    print("Multiply the normalized prices by the allocations and then by the starting capital. Sum these values across all stocks for each day to get the total daily portfolio value.")
    print()
    print("Expected Output (Daily Portfolio Value):")
    port_val = get_portfolio_value(prices, allocations, start_val)
    print("get_portfolio_value() returns:")
    print(port_val)
    print()

    # TODO: Assess the portfolio (students should implement the missing parts)
    cr, adr, sddr, sr, ev, beta, alpha, optional_stats \
        = assess_portfolio(
            start_date, end_date,
            symbols,
            allocations,
            sv=start_val,
            rfr=risk_free_rate,
            sf=sample_freq,
            gen_plot=True,
            calc_optional=True
    )