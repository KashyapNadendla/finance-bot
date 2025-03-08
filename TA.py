import requests
import pandas as pd
import pandas_ta as ta

API_KEY = 'your_alpha_vantage_key'

class TechnicalAnalysis:
    def __init__(self, symbol, interval='daily'):
        self.symbol = symbol
        self.interval = 'daily'
        self.api_url = "https://www.alphavantage.co/query"
        self.api_key = "<Your_API_Key>"

    def fetch_alpha_vantage_ta(self, indicator, interval='daily', series_type='close'):
        params = {
            "function": indicator,
            "symbol": self.symbol,
            "interval": interval,
            "series_type": series_type,
            "apikey": self.api_key
        }
        response = requests.get(self.api_url, params=params)
        return response.json()

    def compute_local_ta(self, df):
        df = df.copy()
        import pandas_ta as ta

        # Example indicators:
        df['rsi'] = df['close'].ta.rsi()
        df['macd'] = df['close'].ta.macd()['MACD_12_26_9']

        return df

    def aggregate_ta(self, df):
        ta_summary = {
            "RSI": df['rsi'].iloc[-1],
            "MACD": df['macd'].iloc[-1],
            "SMA_50": df['close'].ta.sma(length=50).iloc[-1],
            "EMA": df['close'].ta.ema().iloc[-1]
        }
        return ta_summary

if __name__ == "__main__":
    ta = TechnicalAnalysis("AAPL")
    df = ta.fetch_daily_data()
    df = ta.compute_additional_indicators(df)
    summary = ta.aggregate_ta(df)
    print(summary)
