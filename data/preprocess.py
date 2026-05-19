import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def calculate_technical_indicators(df):
    """
    Calculates technical indicators on the historical price dataframe.
    Input must contain 'open', 'high', 'low', 'close', 'adj_close', 'volume'.
    """
    df = df.copy()
    
    # Daily returns
    df["daily_return"] = df["adj_close"].pct_change()
    
    # 20-day rolling volatility of daily returns
    df["rolling_volatility"] = df["daily_return"].rolling(window=20).std()
    
    # Moving Average Ratio (10-day EMA / 50-day EMA - 1)
    df["ema_10"] = df["adj_close"].ewm(span=10, adjust=False).mean()
    df["ema_50"] = df["adj_close"].ewm(span=50, adjust=False).mean()
    df["ema_ratio"] = (df["ema_10"] / df["ema_50"]) - 1
    
    # Relative Strength Index (RSI - 14 day)
    delta = df["adj_close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Use exponential moving average for smoothing RSI
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-9)
    df["rsi"] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    ema_12 = df["adj_close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["adj_close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    # Normalise MACD histogram by close price
    df["macd_hist_norm"] = df["macd_hist"] / df["adj_close"]
    
    # Average True Range (ATR - 14 day)
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["adj_close"].shift(1)).abs()
    low_close = (df["low"] - df["adj_close"].shift(1)).abs()
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.rolling(window=14).mean()
    df["natr"] = df["atr"] / df["adj_close"]  # Scale invariant ATR (Normalized)
    
    # Clean up intermediate helper columns
    df = df.drop(columns=["ema_10", "ema_50", "macd", "macd_signal", "macd_hist", "atr"])
    
    return df

def preprocess_data(ticker=config.TICKER):
    """
    Loads raw stock and macro data, aligns timestamps,
    engineers features, fills missing values, and splits into train/test datasets.
    """
    stock_path = config.DATA_DIR / f"{ticker}_raw.csv"
    macro_path = config.DATA_DIR / "macro_raw.csv"
    
    if not stock_path.exists() or not macro_path.exists():
        raise FileNotFoundError("Raw data files not found. Run fetch.py first.")
        
    stock_df = pd.read_csv(stock_path)
    macro_df = pd.read_csv(macro_path)
    
    # Convert dates to datetime
    stock_df["date"] = pd.to_datetime(stock_df["date"])
    macro_df["date"] = pd.to_datetime(macro_df["date"])
    
    # Merge datasets: align macro data with trading days
    # Since macro values (e.g. Fed funds rate, yield spread) might be updated daily/monthly,
    # we merge them on stock dates, keeping all stock dates.
    merged_df = pd.merge(stock_df, macro_df, on="date", how="left")
    
    # Sort chronologically
    merged_df = merged_df.sort_values("date").reset_index(drop=True)
    
    # Handle missing values in macro indicators by forward-filling then backward-filling
    # (e.g., FRED does not report on holidays/weekends, but stock market dates are trading days)
    macro_cols = list(config.FRED_SERIES.keys())
    merged_df[macro_cols] = merged_df[macro_cols].ffill().bfill()
    
    # Engineer market microstructure variables
    np.random.seed(42)
    vix_factor = merged_df["vix"] / 100.0
    noise_spread = np.random.exponential(scale=0.0001, size=len(merged_df))
    # Spread is relative to close price (e.g., 0.02% base + scaling with VIX + noise)
    merged_df["bid_ask_spread"] = 0.0002 + 0.001 * vix_factor + noise_spread
    
    # Order Book Imbalance (OBI) as a mean-reverting process correlated with returns
    daily_returns = merged_df["close"].pct_change().fillna(0.0).values
    obi = np.zeros(len(merged_df))
    rho = 0.3  # mean reversion autocorrelation
    beta = 10.0  # correlation with returns
    for t in range(1, len(merged_df)):
        noise = np.random.normal(loc=0.0, scale=0.15)
        obi[t] = rho * obi[t-1] + (1.0 - rho) * np.tanh(beta * daily_returns[t]) + noise
    merged_df["order_book_imbalance"] = np.clip(obi, -1.0, 1.0)
    
    # Engineer technical indicators
    df_features = calculate_technical_indicators(merged_df)
    
    # Drop rows with NaN values resulting from rolling window indicators (first 20 rows)
    df_features = df_features.dropna().reset_index(drop=True)
    
    # Split into train and test sets based on dates
    train_df = df_features[df_features["date"] < pd.to_datetime(config.END_DATE) - pd.DateOffset(years=2)]
    test_df = df_features[df_features["date"] >= pd.to_datetime(config.END_DATE) - pd.DateOffset(years=2)]
    
    # Save processed files
    processed_path = config.DATA_DIR / f"{ticker}_processed.csv"
    train_path = config.DATA_DIR / f"{ticker}_train.csv"
    test_path = config.DATA_DIR / f"{ticker}_test.csv"
    
    df_features.to_csv(processed_path, index=False)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Preprocessing complete for {ticker}:")
    print(f"  - Full processed dataset: {processed_path.name} (Shape: {df_features.shape})")
    print(f"  - Train set: {train_path.name} (Shape: {train_df.shape})")
    print(f"  - Test set: {test_path.name} (Shape: {test_df.shape})")
    
    return df_features, train_df, test_df

if __name__ == "__main__":
    try:
        preprocess_data()
    except Exception as e:
        print(f"Preprocessing failed: {e}")
