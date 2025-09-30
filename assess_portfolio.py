from statistics import correlation

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

from get_data import get_data

def plot_normalized_data(df, title="Normalized prices", xlabel="Date", ylabel="Normalized price"):
    """Normalize given stock prices and plot for comparison."""
    if df is None: 
        return

    # Handle list of portfolios or single portfolio/DataFrame
    if isinstance(df, list):
        # Multiple portfolios provided
        portfolios_to_plot = []
        labels = []
        
        for i, portfolio in enumerate(df):
            if isinstance(portfolio, pd.Series):
                portfolio_norm = (portfolio / portfolio.iloc[0])
                if i == 0:
                    labels.append('Original Portfolio')
                else:
                    labels.append(f'Portfolio {i+1}')
                portfolios_to_plot.append(portfolio_norm)
        
        # Get dates from first portfolio
        dates = df[0].index
    else:
        # Single portfolio or DataFrame
        if df.empty:
            return
            
        if isinstance(df, pd.Series):
            portfolio_norm = (df / df.iloc[0]).rename('Portfolio')
            portfolios_to_plot = [portfolio_norm]
            labels = ['Portfolio']
        else:
            normed = df / df.iloc[0]
            portfolio_norm = normed.mean(axis=1).rename('Portfolio')
            portfolios_to_plot = [portfolio_norm]
            labels = ['Portfolio']
        
        dates = df.index

    # Get SPY data
    prices_SPY = get_data(['SPY'], dates)
    spy_norm = (prices_SPY['SPY'] / prices_SPY['SPY'].iloc[0]).rename('SPY')

    # Plot portfolios
    ax = None
    for i, (portfolio_norm, label) in enumerate(zip(portfolios_to_plot, labels)):
        if ax is None:
            ax = portfolio_norm.plot(figsize=(12, 8), linewidth=2.0, label=label)
        else:
            portfolio_norm.plot(ax=ax, linewidth=2.0, label=label)
    
    # Add SPY
    spy_norm.plot(ax=ax, linewidth=2.5, color='black', label='SPY')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show() 
    
    return

def get_portfolio_stats(port_val, prices_SPY, rfr=0.0, sf=252.0, calc_optional=False):
    """Calculate portfolio statistics: core and optional metrics."""

    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio, beta, alpha = 0, 0, 0, 0, 0, 0

    cum_ret = (port_val.iloc[-1] / port_val.iloc[0]) - 1

    daily_returns = port_val.pct_change(fill_method=None).dropna()
    market_daily = prices_SPY.pct_change(fill_method=None).dropna()

    # Align portfolio and market returns by date
    aligned = pd.concat([daily_returns, market_daily], axis=1, join='inner')
    aligned.columns = ['portfolio', 'market']
    
    # Remove any remaining NaN values
    aligned = aligned.dropna()
    
    avg_daily_ret = aligned['portfolio'].mean()
    std_daily_ret = aligned['portfolio'].std()

    # Sharpe ratio using excess returns
    excess_daily = aligned['portfolio'] - rfr  # rfr must be daily
    sharpe_ratio = np.sqrt(sf) * (excess_daily.mean() / excess_daily.std())
    
    # Beta = Cov(portfolio, market) / Var(market)
    cov_pm = aligned.cov().loc['portfolio', 'market']
    var_m = aligned['market'].var()
    beta = cov_pm / var_m 
    
    market_excess_mean = (aligned['market'] - rfr).mean()
    expected_port_mean = rfr + beta * market_excess_mean
    alpha = avg_daily_ret - expected_port_mean

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
    # Use business-day date range with DatetimeIndex
    dates = pd.bdate_range(sd, ed)  # Business days only

    # Get stock data and SPY data
    prices = get_data(syms, dates)
    prices_SPY = get_data(['SPY'], dates)

    # Get portfolio value
    port_val = get_portfolio_value(prices, allocs, sv)

    # Get portfolio statistics - call function
    cr, adr, sddr, sr, beta, alpha, optional_stats = get_portfolio_stats(
        port_val,
        prices_SPY['SPY'],
        rfr=rfr,
        sf=sf,
        calc_optional=calc_optional
    )

    # Compute correlation with the market (SPY)
    correlation = port_val.pct_change(fill_method=None).corr(prices_SPY['SPY'].pct_change(fill_method=None))

    # Implement plotting if `gen_plot` is True
    if gen_plot:
        plot_normalized_data(prices, title="Portfolio vs Market (SPY)")
    
    ev = port_val.iloc[-1]

    print_portfolio_stats(cr, adr, sddr, sr, sv, ev, correlation, beta, alpha, optional_stats)
    return cr, adr, sddr, sr, ev, beta, alpha, optional_stats


def print_portfolio_stats(cr, adr, sddr, sr, sv, ev, correlation, beta, alpha, optional_stats):
    """Pretty print portfolio statistics with optional statistics."""
    print(f"Cumulative Return: {cr:.4f}")
    print(f"Avg Daily Return: {adr:.6f}")
    print(f"Volatility (Std Dev): {sddr:.6f}")
    print(f"Sharpe Ratio: {sr:.4f}")
    print(f"{'Correlation with SPY:':<30} {correlation:>12,.4f}")
    print(f"Beta: {beta:.4f}")
    print(f"Alpha: {alpha:.6f}")
    print(f"Start Portfolio Value: ${sv:,.2f}")
    print(f"End Portfolio Value: ${ev:,.2f}")

    if optional_stats:
        print("\nAdditional Statistics:")
        for stat, value in optional_stats.items():
            if value is not None:
                print(f"{stat}: {value:.4f}")
            else:
                print(f"{stat}: Not implemented")


# Testing the function with a specific set of stocks and allocations
if __name__ == "__main__":
    start_date = dt.datetime(2019, 1, 1)
    end_date = dt.datetime(2019, 12, 31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']


    my_port = ['QQQ', 'CRM', 'JPM', 'HD', 'GLD']

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

    prices_SPY = get_data(['SPY'], dates)
    port_val = get_portfolio_value(prices, allocations, start_val)

    # Calculate my_port portfolio
    my_port_allocations = [1.0/len(my_port)] * len(my_port)  # Equal weight
    my_port_prices = get_data(my_port, dates)
    my_port_val = get_portfolio_value(my_port_prices, my_port_allocations, start_val)

    # Plot both portfolios and SPY
    plot_normalized_data([port_val, my_port_val], title="Portfolio Comparison: Original vs My Portfolio vs SPY")

    # Calculate statistics for both portfolios
    stats = get_portfolio_stats(port_val, prices_SPY['SPY'], rfr=risk_free_rate, sf=sample_freq)
    my_stats = get_portfolio_stats(my_port_val, prices_SPY['SPY'], rfr=risk_free_rate, sf=sample_freq)

    
    # Unpack stats for pretty printing
    cr, adr, sddr, sr, beta, alpha, optional_stats = stats
    correlation = port_val.pct_change(fill_method=None).corr(prices_SPY['SPY'].pct_change(fill_method=None))
    ev = port_val.iloc[-1]
    print("=== FORMATTED ORIGINAL PORTFOLIO STATISTICS ===")
    print_portfolio_stats(cr, adr, sddr, sr, start_val, ev, correlation, beta, alpha, optional_stats)
    print()
    
    
    # Unpack my_stats for pretty printing
    my_cr, my_adr, my_sddr, my_sr, my_beta, my_alpha, my_optional_stats = my_stats
    my_correlation = my_port_val.pct_change(fill_method=None).corr(prices_SPY['SPY'].pct_change(fill_method=None))
    my_ev = my_port_val.iloc[-1]
    print("=== FORMATTED MY PORTFOLIO STATISTICS ===")
    print_portfolio_stats(my_cr, my_adr, my_sddr, my_sr, start_val, my_ev, my_correlation, my_beta, my_alpha, my_optional_stats)
    print()
    
    # Step 1.2: Calculate Daily Portfolio Value
    print("Step 1.2: Calculate Daily Portfolio Value")
    print("Multiply the normalized prices by the allocations and then by the starting capital. Sum these values across all stocks for each day to get the total daily portfolio value.")
    print()
    print("Expected Output (Daily Portfolio Value):")
    print("get_portfolio_value() returns:")
    print(port_val)
    print()