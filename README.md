# Personal Finance Assistant Bot

## Abstract

The Personal Finance Assistant Bot is an AI-driven platform designed to empower users with effective personal finance management tools. Leveraging OpenAI's GPT-4 model, LangChain, and real-time financial data, this bot provides personalized financial advice, asset tracking, and live market insights. Key features include a conversational chatbot, real-time stock performance tracking, daily financial news, interactive budgeting tools, and a Retrieval-Augmented Generation (RAG) system for tailored insights using uploaded financial documents.

## Objective

### Key Objectives:
1. **Financial Knowledge Empowerment**: Enable informed financial decision-making with personalized insights.
2. **Real-Time Market Insights**: Provide live stock data and financial news for timely updates.
3. **Interactive Tools for Finance Management**: Support effective tracking of expenses, savings, and investments.
4. **Enhanced User Experience**: Build an intuitive platform for easy personal finance management.


## Implementation

### Technologies Used
- **Programming Language**: Python
- **Framework**: Streamlit for interactive web app development
- **APIs and Libraries**:
  - **OpenAI API**: For generating personalized financial responses.
  - **LangChain (RAG)**: Processes user-uploaded documents for context-aware responses.
  - **Yahoo Finance (`yfinance`)**: Fetches live stock prices and historical data.
  - **NewsAPI**: Retrieves trending financial news.
  - **PyPDF2**: Processes PDF documents for custom insights.
  - **Chroma Vector Store**: Manages document-based embeddings for RAG.
  - **CoinGecko API**: Provides cryptocurrency pricing.
- **Data Visualization**: Streamlit's charting and Matplotlib.
- **Environment Management**: `dotenv` for secure environment variable handling.

### Key Components

1. **Financial Data Input**: Users provide financial details (e.g., income, expenses) to personalize recommendations.
2. **Document Upload and RAG**: Users upload PDFs like statements or plans, which the bot processes to provide specific, context-driven insights.
3. **Live Market Data**: Real-time stock prices are available through the asset tracker, including historical charts and alerts for price changes.
4. **Interactive Chatbot**: The GPT-4-powered chatbot responds to finance queries using user data, RAG insights, and live market info.
5. **Financial News Updates**: Displays top financial news to keep users informed about market trends.
6. **Financial Tools**: Includes budgeting and expense tracking to calculate savings and manage spending.

## Impact

The bot aims to:
- **Empower Financial Decision-Making**: Provides data-driven recommendations aligned with user goals.
- **Increase User Engagement**: Real-time updates and interactive tools keep users actively managing their finances.
- **Foster Financial Education**: Simplifies complex financial concepts for better financial literacy.
- **Enhance User Confidence**: Encourages informed financial management for improved financial health.


## Features

### 1. **Financial Details Input**
- **Purpose**: To personalize recommendations based on user-provided data.
- **Functionality**: Users input savings, expenses, and goals, which inform tailored chatbot responses.

### 2. **PDF Upload with RAG**
- **Purpose**: Customizes insights using user-uploaded financial documents.
- **Functionality**: Extracts data from PDFs (e.g., statements), which feeds into the RAG system for relevant, user-specific responses.

### 3. **Daily Financial News**
- **Purpose**: Keeps users informed of market developments.
- **Functionality**: Displays top daily news headlines, clickable for full articles.

### 4. **Asset Tracker with Real-Time Data**
- **Purpose**: Enables users to monitor asset performance with live data.
- **Functionality**: Displays real-time stock prices, historical charts, and alerts for significant price changes.

### 5. **Interactive Chatbot with RAG**
- **Purpose**: Provides personalized responses using real-time data and document-based insights.
- **Functionality**: The chatbot integrates user financial data, live stock info, and uploaded documents for well-rounded advice.

### 6. **Budgeting and Savings Tools**
- **Purpose**: Assists with tracking monthly savings and expenses.
- **Functionality**: Users input income and expenses; the tool calculates and displays monthly savings or deficits.


## Future Improvements

1. **Expanded Asset Coverage**: Include foreign exchanges and additional asset classes.
2. **Advanced Data Visualization**: Add a news ticker and customizable dashboards.
3. **Enhanced Personalization**: Support multi-language options and secure user accounts for saved preferences.
4. **Macro Financial Monitoring**: Display economic indicators and market sentiment analysis.
5. **Financial Education**: Offer tutorials and live webinars for skill-building in personal finance.


## Conclusion

The Personal Finance Assistant Bot combines personalized advice, live market data, and interactive financial tools to empower users in managing their finances. By leveraging RAG, real-time stock updates, and AI-powered insights, the bot simplifies complex finance topics, making financial management more accessible and effective.

*This project highlights the power of AI in transforming financial literacy and accessibility, contributing to healthier financial habits and increased user engagement.* 
