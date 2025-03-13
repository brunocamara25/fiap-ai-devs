import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

# Page config
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📊 Genetic Algorithm Portfolio Optimizer")
st.markdown("""
This app uses a genetic algorithm to find the optimal portfolio allocation based on the Sharpe Ratio.
You can select stocks, set your investment amount, and watch the optimization process in real-time!
""")

# Sidebar
with st.sidebar:
    st.header("📝 Parameters")
    
    # Investment amount
    investment = st.number_input(
        "Investment Amount ($)",
        min_value=1000,
        max_value=10000000,
        value=10000,
        step=1000
    )
    
    # Date range
    st.subheader("Date Range")
    start_date = st.date_input(
        "Start Date",
        value=pd.Timestamp("2020-01-01")
    )
    end_date = st.date_input(
        "End Date",
        value=pd.Timestamp("2023-01-01")
    )
    
    # Stock selection
    st.subheader("Stock Selection")
    default_tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    custom_tickers = st.text_input(
        "Enter additional stock tickers (comma-separated)",
        "META, NVDA"
    ).replace(" ", "")
    
    all_tickers = default_tickers + [t.strip() for t in custom_tickers.split(",") if t.strip()]
    selected_tickers = st.multiselect(
        "Select stocks for your portfolio",
        all_tickers,
        default=default_tickers[:3]
    )
    
    # Algorithm parameters
    st.subheader("Algorithm Parameters")
    population_size = st.slider("Population Size", 50, 200, 100)
    num_generations = st.slider("Number of Generations", 10, 100, 50)
    mutation_rate = st.slider("Mutation Rate", 0.0, 0.5, 0.1)
    risk_free_rate = st.slider("Risk-Free Rate (%)", 0.0, 5.0, 2.0) / 100

def download_data(tickers, start_date, end_date):
    """Download stock data"""
    try:
        data = yf.download(tickers, start=start_date, end=end_date,auto_adjust=False)['Adj Close']
        if isinstance(data, pd.Series):
            data = pd.DataFrame(data)
        return data
    except Exception as e:
        st.error(f"Error downloading data: {str(e)}")
        return None

def calculate_metrics(weights, returns, cov_matrix, risk_free_rate):
    """Calculate portfolio metrics"""
    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_vol
    return portfolio_return, portfolio_vol, sharpe_ratio

def create_individual(size):
    """Create random portfolio weights"""
    weights = np.random.random(size)
    return weights / np.sum(weights)

def optimize_portfolio():
    if len(selected_tickers) < 2:
        st.warning("Please select at least 2 stocks.")
        return
    
    # Download data
    with st.spinner("Downloading stock data..."):
        data = download_data(selected_tickers, start_date, end_date)
        if data is None:
            return
    
    # Calculate returns and covariance
    returns = data.pct_change().dropna()
    cov_matrix = returns.cov()
    
    # Display current stock prices
    latest_prices = data.iloc[-1]
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Current Stock Prices")
        price_df = pd.DataFrame({
            'Stock': latest_prices.index,
            'Price': latest_prices.values
        })
        st.dataframe(price_df.style.format({'Price': '${:.2f}'}))
    
    with col2:
        st.subheader("Stock Returns")
        returns_chart = st.empty()
        fig_returns, ax = plt.subplots(figsize=(10, 6))
        (returns + 1).cumprod().plot(ax=ax)
        ax.set_title("Cumulative Returns")
        ax.grid(True)
        returns_chart.pyplot(fig_returns)
        plt.close(fig_returns)
    
    # Setup progress displays
    progress_bar = st.progress(0)
    status_text = st.empty()
    metrics_text = st.empty()
    
    # Charts
    col3, col4 = st.columns(2)
    with col3:
        progress_chart = st.empty()
    with col4:
        allocation_chart = st.empty()
    
    # Initialize population
    population_size = 100
    population = [create_individual(len(selected_tickers)) for _ in range(population_size)]
    best_sharpe = float('-inf')
    best_weights = None
    best_history = []
    
    # Optimization
    fig_progress, ax_progress = plt.subplots(figsize=(10, 6))
    fig_allocation, ax_allocation = plt.subplots(figsize=(10, 6))
    
    try:
        for generation in range(num_generations):
            # Evaluate fitness
            fitness_scores = []
            for weights in population:
                portfolio_return, portfolio_vol, sharpe_ratio = calculate_metrics(
                    weights, returns, cov_matrix, risk_free_rate
                )
                fitness_scores.append(sharpe_ratio)
            
            # Update best solution
            max_idx = np.argmax(fitness_scores)
            if fitness_scores[max_idx] > best_sharpe:
                best_sharpe = fitness_scores[max_idx]
                best_weights = population[max_idx].copy()
            best_history.append(best_sharpe)
            
            # Update progress
            progress = (generation + 1) / num_generations
            progress_bar.progress(progress)
            status_text.text(f"Generation {generation + 1}/{num_generations}")
            
            # Update metrics
            if best_weights is not None:
                ret, vol, _ = calculate_metrics(best_weights, returns, cov_matrix, risk_free_rate)
                metrics_text.markdown(f"""
                **Current Best Portfolio:**
                - Expected Annual Return: {ret:.2%}
                - Expected Annual Volatility: {vol:.2%}
                - Sharpe Ratio: {best_sharpe:.4f}
                """)
            
            # Update charts
            if generation % 2 == 0:
                # Progress chart
                ax_progress.clear()
                ax_progress.plot(best_history, 'b-')
                ax_progress.set_xlabel('Generation')
                ax_progress.set_ylabel('Best Sharpe Ratio')
                ax_progress.set_title('Optimization Progress')
                ax_progress.grid(True)
                progress_chart.pyplot(fig_progress)
                
                # Allocation chart
                ax_allocation.clear()
                if best_weights is not None:
                    ax_allocation.pie(best_weights, labels=selected_tickers, autopct='%1.1f%%')
                    ax_allocation.set_title('Current Best Portfolio Allocation')
                allocation_chart.pyplot(fig_allocation)
            
            # Selection
            new_population = []
            while len(new_population) < population_size:
                # Tournament selection
                tournament_size = 3
                tournament = np.random.choice(population_size, tournament_size)
                parent1 = population[tournament[np.argmax([fitness_scores[i] for i in tournament])]]
                tournament = np.random.choice(population_size, tournament_size)
                parent2 = population[tournament[np.argmax([fitness_scores[i] for i in tournament])]]
                
                # Crossover
                if np.random.random() < 0.8:
                    alpha = np.random.random()
                    child = alpha * parent1 + (1 - alpha) * parent2
                    child = child / np.sum(child)
                else:
                    child = parent1.copy()
                
                # Mutation
                if np.random.random() < mutation_rate:
                    mutation = np.random.normal(0, 0.1, len(selected_tickers))
                    child = child + mutation
                    child = np.clip(child, 0, 1)
                    child = child / np.sum(child)
                
                new_population.append(child)
            
            population = new_population
    
    except Exception as e:
        st.error(f"Optimization error: {str(e)}")
        return
    finally:
        plt.close(fig_progress)
        plt.close(fig_allocation)
    
    # Final Results
    st.header("🎯 Final Portfolio")
    
    col5, col6 = st.columns(2)
    
    with col5:
        st.subheader("Portfolio Metrics")
        final_return, final_vol, final_sharpe = calculate_metrics(
            best_weights, returns, cov_matrix, risk_free_rate
        )
        
        # Calculate projected returns for different time horizons
        monthly_return = (1 + final_return) ** (1/12) - 1
        projected_values = {
            "1 month": investment * (1 + monthly_return),
            "3 months": investment * (1 + monthly_return) ** 3,
            "6 months": investment * (1 + monthly_return) ** 6,
            "1 year": investment * (1 + final_return),
            "2 years": investment * (1 + final_return) ** 2,
            "5 years": investment * (1 + final_return) ** 5
        }
        
        st.markdown(f"""
        #### Current Metrics
        - Expected Annual Return: **{final_return:.2%}**
        - Expected Annual Volatility: **{final_vol:.2%}**
        - Sharpe Ratio: **{final_sharpe:.4f}**
        
        #### Projected Investment Value
        Assuming the expected return rate remains constant:
        """)
        
        for period, value in projected_values.items():
            profit = value - investment
            profit_percent = (profit / investment) * 100
            st.markdown(f"""
            **{period}:**
            - Value: **${value:,.2f}**
            - Profit: **${profit:,.2f}** (*{profit_percent:+.1f}%*)
            """)
            
        st.warning("""
        ⚠️ Note: These projections are based on historical data and expected returns. 
        Actual returns may vary due to market conditions and other factors.
        Past performance does not guarantee future results.
        """)
    
    with col6:
        st.subheader("Investment Allocation")
        allocation_df = pd.DataFrame({
            'Stock': selected_tickers,
            'Weight': best_weights,
            'Amount': best_weights * investment,
            'Shares': (best_weights * investment / latest_prices).round(2)
        })
        st.dataframe(
            allocation_df.style.format({
                'Weight': '{:.2%}',
                'Amount': '${:,.2f}',
                'Shares': '{:.2f}'
            })
        )

# Run button
if st.button("🚀 Optimize Portfolio"):
    optimize_portfolio()
