# import requests
# import pandas as pd
# import pandas_ta as ta

# API_KEY = 'your_alpha_vantage_key'

# class TechnicalAnalysis:
#     def __init__(self, symbol, interval='daily'):
#         self.symbol = symbol
#         self.interval = 'daily'
#         self.api_url = "https://www.alphavantage.co/query"
#         self.api_key = "<Your_API_Key>"

#     def fetch_alpha_vantage_ta(self, indicator, interval='daily', series_type='close'):
#         params = {
#             "function": indicator,
#             "symbol": self.symbol,
#             "interval": interval,
#             "series_type": series_type,
#             "apikey": self.api_key
#         }
#         response = requests.get(self.api_url, params=params)
#         return response.json()

#     def compute_local_ta(self, df):
#         df = df.copy()
#         import pandas_ta as ta

#         # Example indicators:
#         df['rsi'] = df['close'].ta.rsi()
#         df['macd'] = df['close'].ta.macd()['MACD_12_26_9']

#         return df

#     def aggregate_ta(self, df):
#         ta_summary = {
#             "RSI": df['rsi'].iloc[-1],
#             "MACD": df['macd'].iloc[-1],
#             "SMA_50": df['close'].ta.sma(length=50).iloc[-1],
#             "EMA": df['close'].ta.ema().iloc[-1]
#         }
#         return ta_summary

# if __name__ == "__main__":
#     ta = TechnicalAnalysis("AAPL")
#     df = ta.fetch_daily_data()
#     df = ta.compute_additional_indicators(df)
#     summary = ta.aggregate_ta(df)
#     print(summary)
# TA.py
import requests
import pandas as pd
import numpy as np
import pandas_ta as ta
import os
from dotenv import load_dotenv
load_dotenv()

class TechnicalAnalysis:
    def __init__(self, symbol, interval='daily'):
        self.symbol = symbol
        self.interval = interval
        self.api_url = "https://www.alphavantage.co/query"
        self.api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not self.api_key:
            raise ValueError("Alpha Vantage API key not found in environment variables")

    def fetch_daily_data(self, outputsize="compact"):
        """
        Fetch daily historical price data from Alpha Vantage.
        """
        try:
            params = {
                "function": "TIME_SERIES_DAILY",
                "symbol": self.symbol,
                "outputsize": outputsize,  # compact or full
                "apikey": self.api_key
            }
            response = requests.get(self.api_url, params=params)
            if response.status_code != 200:
                print(f"Error fetching data: {response.status_code} - {response.text}")
                return pd.DataFrame()
            data = response.json()
            if "Error Message" in data:
                print(f"API Error: {data['Error Message']}")
                return pd.DataFrame()
            time_series = data.get("Time Series (Daily)", {})
            if not time_series:
                print("No time series data returned")
                return pd.DataFrame()
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            for col in df.columns:
                df[col] = pd.to_numeric(df[col])
            return df
        except Exception as e:
            print(f"Exception when fetching daily data for {self.symbol}: {e}")
            return pd.DataFrame()

    def fetch_alpha_vantage_ta(self, indicator, time_period=14, series_type='close'):
        """
        Fetch technical indicators directly from Alpha Vantage.
        """
        try:
            function_map = {
                "rsi": "RSI",
                "macd": "MACD",
                "sma": "SMA",
                "ema": "EMA",
                "bbands": "BBANDS",
                "stoch": "STOCH"
            }
            function = function_map.get(indicator.lower())
            if not function:
                print(f"Unsupported indicator: {indicator}")
                return None
            params = {
                "function": function,
                "symbol": self.symbol,
                "interval": self.interval,
                "time_period": time_period,
                "series_type": series_type,
                "apikey": self.api_key
            }
            # Adjust parameters for MACD and BBANDS
            if function == "MACD":
                params.pop("time_period", None)
                params.update({"fastperiod": "12", "slowperiod": "26", "signalperiod": "9"})
            if function == "BBANDS":
                params.update({"nbdevup": "2", "nbdevdn": "2", "matype": "0"})
            response = requests.get(self.api_url, params=params)
            if response.status_code != 200:
                print(f"Error fetching {indicator}: {response.status_code} - {response.text}")
                return None
            data = response.json()
            if "Error Message" in data:
                print(f"API Error for {indicator}: {data['Error Message']}")
                return None
            # Process response based on indicator
            key_mapping = {
                "RSI": "Technical Analysis: RSI",
                "MACD": "Technical Analysis: MACD",
                "SMA": "Technical Analysis: SMA",
                "EMA": "Technical Analysis: EMA",
                "BBANDS": "Technical Analysis: BBANDS",
                "STOCH": "Technical Analysis: STOCH"
            }
            tech_data = data.get(key_mapping.get(function, ""), {})
            if not tech_data:
                print(f"No {indicator} data returned")
                return None
            return data
        except Exception as e:
            print(f"Exception when fetching {indicator} for {self.symbol}: {e}")
            return None

    def compute_local_ta(self, df):
        """
        Compute technical indicators locally using pandas_ta.
        """
        if df.empty:
            print("Empty dataframe provided. Cannot compute technical indicators.")
            return df
        try:
            df = df.copy()
            df['rsi'] = ta.rsi(df['close'], length=14)
            macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
            df = pd.concat([df, macd], axis=1)
            bbands = ta.bbands(df['close'], length=20, std=2)
            df = pd.concat([df, bbands], axis=1)
            df['sma_20'] = ta.sma(df['close'], length=20)
            df['sma_50'] = ta.sma(df['close'], length=50)
            df['sma_200'] = ta.sma(df['close'], length=200)
            df['ema_12'] = ta.ema(df['close'], length=12)
            df['ema_26'] = ta.ema(df['close'], length=26)
            stoch = ta.stoch(df['high'], df['low'], df['close'], k=14, d=3, smooth_k=3)
            df = pd.concat([df, stoch], axis=1)
            df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            ppo = ta.ppo(df['close'], fast=12, slow=26, signal=9)
            df = pd.concat([df, ppo], axis=1)
            return df
        except Exception as e:
            print(f"Exception when computing technical indicators for {self.symbol}: {e}")
            return df

    def aggregate_ta(self, df):
        """
        Aggregate technical indicators into a summary.
        """
        if df.empty:
            return {"symbol": self.symbol, "error": "No data available"}
        try:
            latest = df.iloc[-1]
            # Determine trend based on SMA50 and SMA200
            if latest.get('sma_50', np.nan) > latest.get('sma_200', np.nan):
                trend = "Bullish (Golden Cross)"
            elif latest.get('sma_50', np.nan) < latest.get('sma_200', np.nan):
                trend = "Bearish (Death Cross)"
            else:
                trend = "Neutral"
            rsi = latest.get('rsi', np.nan)
            if np.isnan(rsi):
                rsi_signal = "Unknown"
            elif rsi < 30:
                rsi_signal = "Oversold"
            elif rsi > 70:
                rsi_signal = "Overbought"
            else:
                rsi_signal = "Neutral"
            macd_line = latest.get('MACD_12_26_9', np.nan)
            macd_signal = latest.get('MACDs_12_26_9', np.nan)
            if np.isnan(macd_line) or np.isnan(macd_signal):
                macd_signal_desc = "Unknown"
            elif macd_line > macd_signal:
                macd_signal_desc = "Bullish"
            else:
                macd_signal_desc = "Bearish"
            bb_lower = latest.get('BBL_20_2.0', np.nan)
            bb_upper = latest.get('BBU_20_2.0', np.nan)
            close = latest.get('close', np.nan)
            if np.isnan(bb_lower) or np.isnan(bb_upper) or np.isnan(close):
                bb_signal = "Unknown"
            elif close < bb_lower:
                bb_signal = "Oversold"
            elif close > bb_upper:
                bb_signal = "Overbought"
            else:
                bb_signal = "Within Bands"
            stoch_k = latest.get('STOCHk_14_3_3', np.nan)
            stoch_d = latest.get('STOCHd_14_3_3', np.nan)
            if np.isnan(stoch_k) or np.isnan(stoch_d):
                stoch_signal = "Unknown"
            elif stoch_k < 20 and stoch_d < 20:
                stoch_signal = "Oversold"
            elif stoch_k > 80 and stoch_d > 80:
                stoch_signal = "Overbought"
            elif stoch_k > stoch_d:
                stoch_signal = "Bullish Crossover"
            else:
                stoch_signal = "Bearish Crossover"
            ta_summary = {
                "symbol": self.symbol,
                "price": latest.get('close', np.nan),
                "rsi": {"value": latest.get('rsi', np.nan), "signal": rsi_signal},
                "macd": {
                    "line": latest.get('MACD_12_26_9', np.nan),
                    "signal": latest.get('MACDs_12_26_9', np.nan),
                    "histogram": latest.get('MACDh_12_26_9', np.nan),
                    "interpretation": macd_signal_desc
                },
                "moving_averages": {
                    "sma20": latest.get('sma_20', np.nan),
                    "sma50": latest.get('sma_50', np.nan),
                    "sma200": latest.get('sma_200', np.nan),
                    "ema12": latest.get('ema_12', np.nan),
                    "ema26": latest.get('ema_26', np.nan),
                    "trend": trend
                },
                "bollinger_bands": {
                    "upper": latest.get('BBU_20_2.0', np.nan),
                    "middle": latest.get('BBM_20_2.0', np.nan),
                    "lower": latest.get('BBL_20_2.0', np.nan),
                    "signal": bb_signal
                },
                "stochastics": {
                    "k": latest.get('STOCHk_14_3_3', np.nan),
                    "d": latest.get('STOCHd_14_3_3', np.nan),
                    "signal": stoch_signal
                },
                "atr": latest.get('atr', np.nan),
                "overall_signal": self._determine_overall_signal(latest)
            }
            return ta_summary
        except Exception as e:
            print(f"Exception when aggregating TA for {self.symbol}: {e}")
            return {"symbol": self.symbol, "error": str(e)}

    def _determine_overall_signal(self, latest):
        """
        Determine overall trading signal based on multiple indicators.
        """
        signals = []
        rsi = latest.get('rsi', np.nan)
        if not np.isnan(rsi):
            signals.append(1 if rsi < 30 else -1 if rsi > 70 else 0)
        macd_line = latest.get('MACD_12_26_9', np.nan)
        macd_signal = latest.get('MACDs_12_26_9', np.nan)
        if not np.isnan(macd_line) and not np.isnan(macd_signal):
            signals.append(1 if macd_line > macd_signal else -1)
        sma50 = latest.get('sma_50', np.nan)
        sma200 = latest.get('sma_200', np.nan)
        close = latest.get('close', np.nan)
        if not np.isnan(sma50) and not np.isnan(close):
            signals.append(1 if close > sma50 else -1)
        if not np.isnan(sma50) and not np.isnan(sma200):
            signals.append(1 if sma50 > sma200 else -1)
        bb_lower = latest.get('BBL_20_2.0', np.nan)
        bb_upper = latest.get('BBU_20_2.0', np.nan)
        if not np.isnan(bb_lower) and not np.isnan(bb_upper) and not np.isnan(close):
            signals.append(1 if close < bb_lower else -1 if close > bb_upper else 0)
        if not signals:
            return "Insufficient Data"
        avg_signal = sum(signals) / len(signals)
        if avg_signal > 0.5:
            return "Strong Buy"
        elif avg_signal > 0:
            return "Buy"
        elif avg_signal < -0.5:
            return "Strong Sell"
        elif avg_signal < 0:
            return "Sell"
        else:
            return "Hold"
