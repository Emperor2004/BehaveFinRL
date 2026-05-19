# BehaveFinRL — Ticker Reference Guide

> A curated list of Indian and US market tickers for training the BehaveFinRL agent.  
> Organised by market, sector, and recommended combinations for optimal regime detection.

---

## Important Note on Alpha Vantage

- Free tier is limited to **25 API requests/day**
- For Indian tickers, **test with a small fetch first** — BSE/NSE coverage can be inconsistent for some symbols
- Verify each ticker is available on your API tier before committing to it in `config.py`

---

## Indian Market Tickers

| Ticker (Alpha Vantage format) | Company / Index | Sector | Why Useful for BehaveFinRL |
|---|---|---|---|
| `NIFTY.NSE` | Nifty 50 Index | Broad Market | Best baseline — full market cycle coverage |
| `SENSEX.BSE` | BSE Sensex | Broad Market | Alternative broad index, corroborates Nifty signals |
| `RELIANCE.BSE` | Reliance Industries | Energy + Retail + Telecom | Largest Indian company, drives index movement |
| `HDFCBANK.BSE` | HDFC Bank | Banking | Most liquid private bank, clean price data |
| `INFY.BSE` | Infosys | IT Services | USD revenue, reacts to global tech sentiment |
| `TCS.BSE` | Tata Consultancy Services | IT Services | Similar to INFY but more stable, good contrast |
| `ICICIBANK.BSE` | ICICI Bank | Banking | Good volatility, different risk profile than HDFC |
| `TATAMOTORS.BSE` | Tata Motors | Auto + EV | High volatility, regime swings are sharp and clear |
| `WIPRO.BSE` | Wipro | IT Services | Lower volatility IT stock, good for Bear regime learning |
| `ONGC.BSE` | Oil & Natural Gas Corp | Energy | Commodity-linked, uncorrelated with IT tickers |
| `SUNPHARMA.BSE` | Sun Pharma | Pharmaceuticals | Defensive sector, useful for Bear regime behaviour |
| `BAJFINANCE.BSE` | Bajaj Finance | NBFC / Fintech | High beta, amplifies market moves — good for HMM |
| `AXISBANK.BSE` | Axis Bank | Banking | More volatile than HDFC, different risk character |
| `MARUTI.BSE` | Maruti Suzuki | Automobile | Consumer demand proxy, rural economy sensitivity |
| `HINDUNILVR.BSE` | Hindustan Unilever | FMCG | Defensive, low volatility — good Bear regime anchor |
| `GOLDBEES.NSE` | Gold BeES ETF | Commodity / Safe Haven | Indian equivalent of GLD — inverse equity relationship |
| `LIQUIDBEES.NSE` | Liquid BeES ETF | Cash Equivalent | Near-zero volatility — useful as safe haven in Bear |

---

## US Market Tickers

| Ticker | Company / Index | Sector | Why Useful for BehaveFinRL |
|---|---|---|---|
| `SPY` | S&P 500 ETF | Broad Market | Most liquid, cleanest regime signals |
| `QQQ` | Nasdaq 100 ETF | Tech-Heavy | High volatility, strong Bull/Bear swings |
| `DIA` | Dow Jones ETF | Blue Chip | Slower moving, good contrast with QQQ |
| `IWM` | Russell 2000 ETF | Small Cap | More sensitive to economic cycles than SPY |
| `GLD` | Gold ETF | Safe Haven | Rises in Bear markets — great regime signal |
| `TLT` | 20+ Year Treasury Bond ETF | Bonds | Inverse equity relationship, key for regime detection |
| `SHY` | Short Term Treasury ETF | Bonds | Near risk-free, anchor for Bear regime |
| `VXX` | VIX Futures ETF | Volatility | Directly tracks fear — spikes in Bear/Volatile regimes |
| `AAPL` | Apple Inc. | Tech | Deep historical data, well-studied price patterns |
| `MSFT` | Microsoft | Tech | More stable than AAPL, good long-term trend data |
| `GOOGL` | Alphabet | Tech + Advertising | Ad revenue sensitivity to economic cycles |
| `JPM` | JPMorgan Chase | Banking | Rate-sensitive, reacts strongly to FRED macro data |
| `XLF` | Financial Sector ETF | Banking / Finance | Broader than individual banks, cleaner signal |
| `XLE` | Energy Sector ETF | Energy | Commodity-linked, low correlation with tech |
| `XLV` | Healthcare Sector ETF | Healthcare | Defensive sector, holds up in Bear regimes |
| `XLU` | Utilities Sector ETF | Utilities | Most defensive sector — classic Bear regime safe haven |
| `BRK.B` | Berkshire Hathaway | Conglomerate | Value-oriented, less volatile than growth stocks |
| `BAC` | Bank of America | Banking | Rate-sensitive like JPM but higher beta |

---

## Recommended Combinations

These combinations are selected for **low inter-ticker correlation** and **distinct regime behaviour** — both critical for clean HMM regime detection and effective Prospect Theory reward shaping.

### Clean Regime Detection (Best for HMM)
```python
TICKERS = ["SPY", "GLD", "TLT"]
```
SPY rises in Bull, GLD rises in Bear, TLT moves inversely to equities.  
Three distinct behavioural profiles — gives the HMM very clear signals to learn from.

---

### India-Focused Research
```python
TICKERS = ["NIFTY.NSE", "GOLDBEES.NSE", "HDFCBANK.BSE"]
```
Broad index + safe haven + banking sector.  
Directly relevant to Indian market dynamics — strong for SIBM context.

---

### Cross-Market Generalisation
```python
TICKERS = ["SPY", "NIFTY.NSE", "GLD", "TLT"]
```
US equity + Indian equity + gold + bonds.  
Teaches the agent cross-market dynamics and currency-regime interactions.

---

### High Volatility Training
```python
TICKERS = ["QQQ", "BAJFINANCE.BSE", "VXX"]
```
High-beta US tech + high-beta Indian fintech + volatility index.  
Sharpens the agent's loss-averse behaviour — maximum Prospect Theory stress-testing.

---

### Defensive vs Aggressive Contrast
```python
TICKERS = ["XLU", "QQQ"]
```
Utilities (most defensive) vs Nasdaq (most aggressive).  
Extreme opposites — creates the clearest possible Bull/Bear regime distinction for HMM.

---

## Sector Coverage Map

```
DEFENSIVE (Bear regime anchors)
├── XLU  — Utilities
├── XLV  — Healthcare
├── GLD  — Gold / Safe Haven
├── TLT  — Long Bonds
├── SHY  — Short Bonds
├── HINDUNILVR.BSE — Indian FMCG
├── SUNPHARMA.BSE  — Indian Pharma
└── LIQUIDBEES.NSE — Indian Cash Equivalent

CYCLICAL (Bull regime leaders)
├── SPY  — US Broad Market
├── QQQ  — US Tech
├── NIFTY.NSE — Indian Broad Market
├── RELIANCE.BSE  — Indian Conglomerate
├── BAJFINANCE.BSE — Indian Fintech
└── TATAMOTORS.BSE — Indian Auto + EV

RATE SENSITIVE (Macro / FRED interaction)
├── TLT  — Long Bonds (inverse rate)
├── JPM  — US Banking
├── XLF  — US Financial Sector
├── HDFCBANK.BSE  — Indian Banking
└── ICICIBANK.BSE — Indian Banking

VOLATILITY / FEAR
└── VXX  — VIX Futures (spikes in Bear/Volatile regimes)
```

---

## Suggested Training Date Range

| Period | Market Events Covered |
|---|---|
| `2018-01-01` to `2019-12-31` | US–China trade war volatility |
| `2020-01-01` to `2020-12-31` | COVID crash + rapid Bull recovery |
| `2021-01-01` to `2021-12-31` | Post-COVID Bull run |
| `2022-01-01` to `2022-12-31` | Rate hike cycle + Bear market |
| `2023-01-01` to `2024-01-01` | Recovery + AI-driven Bull |

**Recommended full range:**
```python
START_DATE = "2018-01-01"
END_DATE   = "2024-01-01"
```
This covers at least one full Bull market, one Bear market, and one High Volatility period — essential for the HMM to learn all three regimes meaningfully.

---

## Adding a New Ticker

1. Verify availability on Alpha Vantage with a test fetch
2. Add the ticker symbol to `TICKERS` list in `config.py`
3. Check correlation with existing tickers — avoid adding tickers that move identically
4. Re-run `data/fetch.py` and `data/preprocess.py`
5. Re-train the HMM (`regime/hmm.py`) — new tickers change the regime signal
6. Re-train the PPO agent (`train.py`) from scratch

> **Rule of thumb:** If two tickers have correlation > 0.85, they are redundant. Pick one.

---

*Part of the BehaveFinRL project — Symbiosis Institute of Business Management, Pune*  
*Author: Om Narayan Pandit | [@omnarayanpandit](https://github.com/omnarayanpandit)*