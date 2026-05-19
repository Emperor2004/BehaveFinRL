import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def fetch_stock_data(ticker=config.TICKER, start_date=config.START_DATE, end_date=config.END_DATE):
    """
    Fetches daily stock OHLCV data. If ticker is SP500, fetches from FRED.
    Otherwise fetches from Alpha Vantage.
    """
    if ticker == "SP500":
        print(f"Fetching daily stock data for {ticker} from FRED (as the main asset)...")
        # Fetch SP500 close from FRED
        df = fetch_fred_series("SP500", "close", start_date, end_date)
        
        # Clean and fill missing values in close price first
        df["close"] = df["close"].ffill().bfill()
        
        # Create standard OHLCV columns using close price
        df["open"] = df["close"]
        df["high"] = df["close"]
        df["low"] = df["close"]
        df["adj_close"] = df["close"]
        df["volume"] = 1000000.0
        
        # Select and order columns
        df = df[["date", "open", "high", "low", "close", "adj_close", "volume"]]
        
        # Save raw CSV
        output_path = config.DATA_DIR / f"{ticker}_raw.csv"
        df.to_csv(output_path, index=False)
        print(f"Stock data (FRED SP500) saved to {output_path}. Shape: {df.shape}")
        return df

    print(f"Fetching daily stock data for {ticker} from Alpha Vantage...")
    
    # Try regular daily close first (free), then adjusted close
    functions = ["TIME_SERIES_DAILY", "TIME_SERIES_DAILY_ADJUSTED"]
    response_json = None
    success = False
    
    for idx, func in enumerate(functions):
        if idx > 0:
            print("Waiting 5 seconds between Alpha Vantage API queries to avoid rate limits...")
            time.sleep(5)
            
        url = f"https://www.alphavantage.co/query?function={func}&symbol={ticker}&outputsize=full&apikey={config.ALPHA_VANTAGE_KEY}"
        
        try:
            response = requests.get(url)
            data = response.json()
            
            # Check for API rate limit or error messages
            if "Note" in data or "Please consider spreading out your free API requests" in data.get("Information", ""):
                print("Warning: Alpha Vantage API rate limit hit. Waiting 60 seconds...")
                time.sleep(60)
                # Retry once
                response = requests.get(url)
                data = response.json()
            
            if "Time Series (Daily)" in data:
                response_json = data
                success = True
                print(f"Successfully fetched data using {func}.")
                break
            elif "Error Message" in data:
                print(f"Error for function {func}: {data['Error Message']}")
            else:
                print(f"API Response message for {func}: {data.get('Information', data.get('Note', 'Unknown error'))}")
        except Exception as e:
            print(f"Exception during request for {func}: {e}")
            
    if not success or response_json is None:
        raise ValueError(f"Could not fetch stock data from Alpha Vantage for {ticker}. Check API key and rate limits.")
        
    # Parse the daily series data
    time_series = response_json["Time Series (Daily)"]
    records = []
    for date_str, metrics in time_series.items():
        # Alpha Vantage returns strings. Convert to floats/dates.
        row = {
            "date": pd.to_datetime(date_str),
            "open": float(metrics.get("1. open")),
            "high": float(metrics.get("2. high")),
            "low": float(metrics.get("3. low")),
            "close": float(metrics.get("4. close")),
        }
        
        # Check for adjusted close, otherwise use standard close
        if "5. adjusted close" in metrics:
            row["adj_close"] = float(metrics.get("5. adjusted close"))
            row["volume"] = float(metrics.get("6. volume"))
        else:
            row["adj_close"] = float(metrics.get("4. close"))
            row["volume"] = float(metrics.get("5. volume"))
            
        records.append(row)
        
    df = pd.DataFrame(records)
    df = df.sort_values("date").reset_index(drop=True)
    
    # Filter by date range
    df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]
    
    # Save raw CSV
    output_path = config.DATA_DIR / f"{ticker}_raw.csv"
    df.to_csv(output_path, index=False)
    print(f"Stock data saved to {output_path}. Shape: {df.shape}")
    return df

def fetch_fred_series(series_id, series_name, start_date=config.START_DATE, end_date=config.END_DATE):
    """
    Fetches a single macroeconomic time series from FRED.
    """
    print(f"Fetching FRED series {series_id} ({series_name})...")
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={config.FRED_API_KEY}&file_type=json"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        if "error_message" in data:
            raise ValueError(f"FRED API error: {data['error_message']}")
            
        observations = data.get("observations", [])
        records = []
        for obs in observations:
            date_str = obs["date"]
            val_str = obs["value"]
            # FRED values can be "." on holidays or non-reporting days, handle this
            try:
                val = float(val_str)
            except ValueError:
                val = np.nan
            records.append({
                "date": pd.to_datetime(date_str),
                series_name: val
            })
            
        df = pd.DataFrame(records)
        df = df.sort_values("date").reset_index(drop=True)
        # Filter by date range
        df = df[(df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))]
        return df
    except Exception as e:
        print(f"Error fetching FRED series {series_id}: {e}")
        raise

def fetch_macro_indicators(start_date=config.START_DATE, end_date=config.END_DATE):
    """
    Fetches all macroeconomic indicators defined in config.FRED_SERIES from FRED.
    Merges them on date and saves to config.DATA_DIR.
    """
    macro_dfs = []
    
    for name, series_id in config.FRED_SERIES.items():
        # Wait a bit to avoid hitting rate limits too quickly
        time.sleep(1)
        df = fetch_fred_series(series_id, name, start_date, end_date)
        macro_dfs.append(df)
        
    # Merge all macroeconomic dataframes on date
    # Use outer join to keep all dates, we'll clean and forward-fill in preprocessing
    merged_df = macro_dfs[0]
    for df in macro_dfs[1:]:
        merged_df = pd.merge(merged_df, df, on="date", how="outer")
        
    merged_df = merged_df.sort_values("date").reset_index(drop=True)
    
    # Save raw CSV
    output_path = config.DATA_DIR / "macro_raw.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"Macroeconomic data saved to {output_path}. Shape: {merged_df.shape}")
    return merged_df

if __name__ == "__main__":
    # Test script execution
    try:
        stock_df = fetch_stock_data()
        macro_df = fetch_macro_indicators()
        print("Data fetch pipeline completed successfully.")
    except Exception as e:
        print(f"Fetch failed: {e}")
