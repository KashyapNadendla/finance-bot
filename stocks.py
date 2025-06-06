import streamlit as st
import os
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

# ---------------------- ALPHA VANTAGE API CONFIG ---------------------- #
BASE_URL = "https://www.alphavantage.co/query"

# List of stock tickers to track
TICKERS = [
        "AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "META", "BRK.B", "TSM", "TSLA", "AVGO", 
        "LLY", "WMT", "JPM", "V", "UNH", "XOM", "NVO", "ORCL", "MA", "PG", "HD", "COST", 
        "JNJ", "ABBV", "BAC", "NFLX", "KO", "CRM", "SAP", "CVX", "ASML", "MRK", "TMUS", 
        "AMD", "TM", "PEP", "LIN", "AZN", "BABA", "CSCO", "NVS", "WFC", "ACN", "ADBE", 
        "TMO", "MCD", "PM", "SHEL", "ABT", "NOW", "AXP", "MS", "TXN", "GE", "IBM", "QCOM", 
        "CAT", "ISRG", "DHR", "RY", "INTU", "VZ", "GS", "DIS", "AMGN", "PDD", "UBER", "HSBC", 
        "CMCSA", "NEE", "RTX", "ARM", "PFE", "T", "HDB", "UL", "AMAT", "SPGI", "BKNG", "LOW", 
        "TTE", "BLK", "PGR", "BHP", "UNP", "SYK", "BX", "ETN", "SNY", "HON", "SCHW", "LMT", 
        "TJX", "BUD", "ANET", "KKR", "MUFG", "BSX", "VRTX", "C", "COP", "ADP", "PANW", "MDT", 
        "MU", "UPS", "CB", "ADI", "NKE", "FI", "BA", "RIO", "DE", "SBUX", "IBN", "GILD", "MMC", 
        "SONY", "PLD", "BMY", "SHOP", "MELI", "UBS", "AMT", "REGN", "LRCX", "PLTR", "SO", "TD", 
        "ICE", "INTC", "ELV", "MDLZ", "HCA", "KLAC", "DELL", "SHW", "INFY", "ENB", "DUK", "SCCO", 
        "CI", "RELX", "EQIX", "ABNB", "WM", "WELL", "MO", "RACE", "TT", "PBR.A", "PBR", "CTAS", "SMFG", 
        "BN", "MCO", "APO", "ZTS", "GD", "APH", "SNPS", "GEV", "CEG", "CME", "PH", "AON", "CDNS", "SPOT", 
        "ITW", "PYPL", "CL", "BP", "CMG", "BTI", "USB", "MSI", "PNC", "CRWD", "NU", "TRI", "GSK", "TDG", "MAR", 
        "NOC", "SAN", "CP", "CNQ", "ECL", "MRVL", "CVS", "DEO", "APD", "MMM", "CNI", "EOG", "TGT", "BDX", 
        "EQNR", "ORLY", "FDX", "BMO", "FCX", "CARR", "CRH", "MCK", "CSX", "BNS", "WMB", "DASH", "COF", "EPD", 
        "WDAY", "NGG", "NXPI", "AJG", "EMR", "RSG", "ADSK", "AFL", "DLR", "FTNT", "TTD", "CM", "PSA", "ROP", 
        "JD", "MET", "HLT", "TFC", "APP", "NSC", "GM", "BBVA", "TRV", "SLB", "ET", "OKE", "SPG", "RCL", "ITUB", 
        "BK", "KMI", "PCAR", "DHI", "SE", "GWW", "NEM", "MFG", "URI", "ING", "SRE", "O", "MFC", "COIN", "NTES", 
        "FANG", "AEP", "MNST", "AZO", "JCI", "PAYX", "PSX", "CPRT", "MSTR", "ALL", "AMP", "TEAM", "FIS", "AIG", 
        "FICO", "D", "AMX", "MPC", "TRP", "SU", "E", "HMC", "CHTR", "CPNG", "OXY", "CCI", "LHX", "LEN", "ROST", 
        "ALC", "VALE", "TEL", "PWR", "WCN", "BCS", "CMI", "PRU", "MPLX", "SQ", "COR", "FAST", "MPWR", "KMB", 
        "KDP", "MSCI", "AEM", "PEG", "TAK", "HLN", "KVUE", "ODFL", "NDAQ", "DDOG", "PCG", "STZ", "LYG", "VST", 
        "CTVA", "TCOM", "VRT", "FLUT", "F", "EW", "HWM", "VLO", "HES", "LNG", "KHC", "MCHP", "KR", "IT", "SNOW", 
        "GEHC", "EXC", "CBRE", "NWG", "FERG", "EA", "GRMN", "IQV", "ACGL", "OTIS", "VRSK", "IR", "AME", "GLW", 
        "IMO", "DFS", "LVS", "STLA", "GIS", "A", "YUM", "DAL", "IRM", "LULU", "IDXX", "BKR", "MLM", "CTSH", 
        "TRGP", "VMC", "SYY", "ALNY", "HSY", "RMD", "ED", "HPQ", "ABEV", "XEL", "CCEP", "WIT", "GOLD", "EXR", 
        "DD", "VEEV", "DOW", "HEI", "ARES", "VICI", "NUE", "EFX", "ARGX", "AXON", "WAB", "AVB", "MTB", "DB", 
        "HIG", "SLF", "BIDU", "EIX", "HUM", "XYL", "ON", "EL", "CNC", "FMX", "NET", "EBAY", "WPM", "CVE", 
        "WEC", "RJF", "BRO", "ROK", "CSGP", "HEI.A", "WTW", "FITB", "WDS", "CHT", "BCE", "FER", "PPG", 
        "TSCO", "LI", "HUBS", "CCL", "ETR", "ANSS", "TTWO", "ZS", "LYB", "ERIC", "DXCM", "EQR", "FCNCA", 
        "RBLX", "K", "NVR", "FCNCO", "STT", "MTD", "VTR", "TW", "IOT", "BNTX", "LYV", "BEKE", "PHM", "TEF", 
        "ADM", "TPL", "DOV", "UAL", "AWK", "HPE", "BIIB", "KEYS", "TYL", "GPN", "FNV", "CAH", "CDW", "SW",
        "NOK", "IFF", "DECK", "BBD", "DTE", "CVNA", "KB", "VLTO", "GIB", "FTV", "DVN", "STM", "HOOD", "SBAC", 
        "TROW", "BR", "LDOS", "CHD", "PHG", "VOD", "IX", "HAL", "NTAP", "FE", "PBA", "TECK", "CQP", "PPL", 
        "TU", "NTR", "ERIE", "ILMN", "CCJ", "BAH", "ES", "HUBB", "AEE", "WY", "CPAY", "ZM", "WDC", "EQT", 
        "HBAN", "GDDY", "QSR", "ROL", "WST", "BAM", "PTC"]

def debug_log(message):
    """Helper to log debug messages to the console."""

def get_api_key():
    """Retrieve the Alpha Vantage API key from the environment."""
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        st.error("Alpha Vantage API key not found in environment variables.")
    return api_key

# =============================================================================
# 1) FETCH DAILY PRICE (MOST RECENT CLOSE + PRICE CHANGE)
# =============================================================================
def fetch_stock_price(ticker):
    """
    Fetch the latest daily stock price for a given ticker using Alpha Vantage.
    Returns a dict with Ticker, Current Price, and Price Change (Today).
    """
    API_KEY = get_api_key()
    if not API_KEY:
        return None


    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "apikey": API_KEY
    }
    try:
        response = requests.get(BASE_URL, params=params)
        data = response.json()

        debug_log(f"Alpha Vantage Daily response for {ticker}:\n{data}")

        if "Time Series (Daily)" not in data:
            st.warning(f"Could not retrieve daily price for {ticker}. Possible API issue, invalid key, or rate limiting.")
            return None

        daily_series = data["Time Series (Daily)"]
        latest_day = max(daily_series.keys())
        day_data = daily_series[latest_day]

        open_price = float(day_data["1. open"])
        close_price = float(day_data["4. close"])
        price_change = ((close_price - open_price) / open_price) * 100

        return {
            "Ticker": ticker,
            "Current Price": f"${close_price:.2f}",
            "Price Change (Today)": f"{price_change:.2f}%"
        }

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching daily stock price for {ticker}: {e}")
        return None

# =============================================================================
# 2) FETCH COMPANY OVERVIEW (DIVIDEND YIELD, ETC.)
# =============================================================================
def fetch_stock_overview(ticker):
    """
    Fetch company overview (dividend yield, market cap, etc.) using Alpha Vantage.
    Returns a dict with "Dividend Yield" key or None on failure.
    """
    API_KEY = get_api_key()
    if not API_KEY:
        return None

    params = {
        "function": "OVERVIEW",
        "symbol": ticker,
        "apikey": API_KEY
    }
    try:
        response = requests.get(BASE_URL, params=params)
        data = response.json()

        debug_log(f"Alpha Vantage Overview response for {ticker}:\n{data}")

        if "Symbol" not in data:
            return None

        dividend_yield = data.get("DividendYield")
        # Check if dividend_yield is None, empty, or not convertible
        if dividend_yield in (None, "", "None"):
            dividend_yield = "N/A"
        else:
            try:
                dividend_yield = f"{float(dividend_yield) * 100:.2f}%"
            except ValueError:
                dividend_yield = "N/A"

        return {"Dividend Yield": dividend_yield}

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching stock overview for {ticker}: {e}")
        return None

# =============================================================================
# 3) COMBINE PRICE + OVERVIEW DATA
# =============================================================================
def fetch_full_stock_data(ticker):
    """
    Fetch both daily price info and company overview, merge into a single dict.
    """
    price_data = fetch_stock_price(ticker)
    if not price_data:
        return None

    overview_data = fetch_stock_overview(ticker)
    if overview_data:
        price_data.update(overview_data)

    return price_data

# =============================================================================
# 4) FETCH ALL TICKERS (IN PARALLEL)
# =============================================================================
def get_asset_data():
    """
    Fetch stock data for all TICKERS using multithreading.
    Returns a list of dicts, one per stock, with keys:
      Ticker, Current Price, Price Change (Today), Dividend Yield
    """
    asset_data = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_full_stock_data, t): t for t in TICKERS}
        for future in as_completed(futures):
            result = future.result()
            if result:
                asset_data.append(result)

    if not asset_data:
        st.warning("No stock data retrieved. You may be rate-limited or have an invalid API key.")
    return asset_data

# =============================================================================
# 5) HELPER: GET FULL HISTORICAL DAILY DATA FOR CHART
# =============================================================================
def fetch_historical_data(ticker):
    """
    Fetch full daily historical data from Alpha Vantage for charting.
    Returns a pandas DataFrame with columns: ['Open', 'High', 'Low', 'Close', 'Volume'],
    indexed by date.
    """
    API_KEY = get_api_key()
    if not API_KEY:
        st.error("Alpha Vantage API key not found. Can't fetch historical data.")
        return pd.DataFrame()

    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "apikey": API_KEY
    }
    try:
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        debug_log(f"Alpha Vantage Historical data for {ticker}:\n{data}")

        if "Time Series (Daily)" not in data:
            st.warning(f"Could not retrieve daily series for chart: {ticker}")
            return pd.DataFrame()

        daily_series = data["Time Series (Daily)"]
        records = []
        for date_str, values in daily_series.items():
            records.append({
                "Date": date_str,
                "Open": float(values["1. open"]),
                "High": float(values["2. high"]),
                "Low": float(values["3. low"]),
                "Close": float(values["4. close"]),
                "Volume": float(values["5. volume"])
            })

        df = pd.DataFrame(records)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)
        return df

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching historical data for {ticker}: {e}")
        return pd.DataFrame()

# =============================================================================
# 6) STREAMLIT DISPLAY FUNCTIONS
# =============================================================================
def display_assets():
    """
    Displays the fetched stock data in a table with an option to select 'preferred assets'.
    """
    st.header("Asset Data")

    if "preferred_assets" not in st.session_state:
        st.session_state["preferred_assets"] = []

    if not st.session_state.get("asset_data"):
        st.warning("Asset data not loaded. Click 'Update Stock Prices' to load.")
        return

    df = pd.DataFrame(st.session_state["asset_data"])
    st.dataframe(df)

    if "Ticker" in df.columns:
        tickers = df["Ticker"].tolist()
        selected_assets = st.multiselect(
            "Select your preferred assets",
            tickers,
            default=st.session_state["preferred_assets"]
        )
        st.session_state["preferred_assets"] = selected_assets

        if selected_assets:
            st.write("Your preferred assets:")
            preferred_df = df[df["Ticker"].isin(selected_assets)]
            st.dataframe(preferred_df)
            check_price_alerts()
    else:
        st.info("No Ticker data available.")

def check_price_alerts():
    """
    If user has selected assets, allow setting a price change threshold
    and display warnings for large daily changes.
    """
    if not st.session_state.get("preferred_assets"):
        return

    alert_threshold = st.slider(
        "Set price change alert threshold (%)",
        min_value=0.0, max_value=10.0, value=5.0
    )

    for asset in st.session_state["asset_data"]:
        if asset["Ticker"] in st.session_state["preferred_assets"]:
            price_change_str = asset["Price Change (Today)"].strip("%")
            try:
                price_change_val = float(price_change_str)
                if abs(price_change_val) >= alert_threshold:
                    st.warning(f"🚨 {asset['Ticker']} changed {price_change_val:.2f}% today!")
            except ValueError:
                pass

def display_asset_charts():
    """
    Let user pick a ticker from the fetched asset data, then fetch daily series from Alpha Vantage
    and display a line chart of the Close prices.
    """
    st.subheader("Asset Price Chart")

    if not st.session_state.get("asset_data"):
        st.info("Asset data not loaded. Please update stock prices to view charts.")
        return

    tickers = [stock["Ticker"] for stock in st.session_state["asset_data"]]
    selected_ticker = st.selectbox("Select a ticker to view price chart:", tickers)

    if selected_ticker:
        with st.spinner("Fetching historical data..."):
            hist_df = fetch_historical_data(selected_ticker)

        if hist_df.empty:
            st.info("No chart data available for the selected ticker.")
        else:
            st.line_chart(hist_df["Close"])


def scout_assets(ticker_list):
    """
    Fetch stock data for the given tickers.
    This function calls get_asset_data() and then filters the results.
    """
    all_data = get_asset_data()
    # Filter the asset data for tickers in the provided list.
    filtered = [asset for asset in all_data if asset.get("Ticker") in ticker_list]
    return filtered

def format_asset_suggestions(asset_data):
    """
    Format asset data suggestions for presentation.
    """
    if not asset_data:
        return "No asset data available."
    suggestions = ""
    for asset in asset_data:
        ticker = asset.get("Ticker", "N/A")
        price = asset.get("Current Price", "N/A")
        change = asset.get("Price Change (Today)", "N/A")
        suggestions += f"{ticker}: {price} ({change})\n"
    return suggestions
