# Personal Finance Assistant Bot

## Overview

This project is a multi-agent finance assistant that integrates real-time financial data (stocks, cryptocurrencies, commodities, and news) with both macroeconomic and technical analysis to provide personalized investment advice. The system leverages multiple Large Language Models (LLMs) to process user input, generate recommendations, and refine those recommendations based on live data and technical indicators.

## Workflow & Ideology

1. **Data Ingestion & Caching**  
   - **Stock Data:** Fetched from Alpha Vantage and cached in a CSV file (with a time-to-live of 10 minutes).  
   - **News & Market Data:** Real-time news is retrieved via NewsAPI; cryptocurrency data is fetched from CoinMarketCap; commodity prices are obtained using Yahoo Finance.
   - **Document Processing:** PDF documents are processed and converted into a vector store for additional context when needed.

2. **User Interaction**  
   - The user provides financial details and queries through a chat interface and forms.
   - Users can also upload their own documents or process existing ones for enhanced context.

3. **Multi-Agent Chain of LLMs**  
   - **Agent 1 (Analysis Agent):** Analyzes the user query to extract investment goals, risk appetite, and expected returns.
   - **Agent 2 (Recommendation Agent):** Uses live market data (stocks, crypto, commodities, news) to generate asset recommendations.
   - **Agent 3 (Evaluation Agent):** Evaluates the recommendations against:
     - **Macroeconomic Conditions:** Factors such as US interest rates, the 10Y Treasury yield, DXY trends, and geopolitical events.
     - **Technical Analysis:** Incorporates key technical indicators (RSI, moving averages, Bollinger Bands) and looks for technical patterns (e.g., falling wedge, head and shoulders, double tops/bottoms, Fibonacci extensions) on both daily and weekly timeframes.
   - The evaluation may iterate (up to 3 times) to adjust recommendations if market conditions warrant a more conservative approach.

4. **Technical Analysis Module**  
   - Utilizes the `ta` library to compute:
     - **RSI (14-period)**
     - **20-day Simple Moving Average (SMA20)**
     - **Bollinger Bands (20-day window, 2 standard deviations)**
   - Analysis is performed on both daily and weekly timeframes to provide a comprehensive view.
   - The results help determine which asset charts are technically stronger, adding another layer of decision-making to the recommendations.

5. **Final Output**  
   - The assistant provides a final, refined recommendation that considers user input, live market data, macroeconomic conditions, and technical analysis.

## Features

- **Real-Time Financial Data:** Live updates for stocks, crypto, commodities, and news.
- **Technical Analysis:** Automated computation of key technical indicators and patterns.
- **Multi-Agent LLM Workflow:** A chain of LLMs that process, recommend, and refine investment advice.
- **User-Friendly Interface:** Built with Streamlit, featuring chat interfaces, data visualization, and interactive forms.
- **Document Processing:** Capability to upload and process PDFs for additional contextual insights.

## Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone <https://github.com/KashyapNadendla/finance-bot>
   cd finance-bot

2. **Install Dependencies:**
  ```bash
  pip install -r requirements.txt
  ```

3. **Configure Environment Variables:**
  Create a .env file in the project root with the following keys:
  ```
  OPENAI_API_KEY=your_openai_api_key
  NEWS_API_KEY=your_news_api_key
  CMC_API_KEY=your_coinmarketcap_api_key
  ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
  ```

4. **Run the Application:**
  ```bash
  streamlit run app.py
  ```

## Usage
- Chat Interface: Interact with the assistant through a conversational interface to ask questions and receive investment advice.
- Asset Data & Charts: View real-time asset data, update stock prices, and see price charts.
- Agentic Advisor: Submit an investment query to receive multi-layered recommendations that incorporate macroeconomic and technical analysis.
- Document Upload & Processing: Upload your own PDF documents to enhance the context for recommendations.
- Budgeting Tool: Manage monthly income and expenses with an integrated budgeting calculator.

## Future Enhancements
- Advanced pattern recognition algorithms for technical chart patterns.
- Integration of additional data sources and asset classes.
- Improved modularity and error handling for even greater robustness.