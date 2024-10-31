# Personal Finance Assistant Bot

## Abstract

The Personal Finance Assistant Bot is an interactive platform designed to empower individuals in managing their finances effectively. Leveraging advanced technologies such as OpenAI's GPT-4 architecture, LangChain for natural language processing, and real-time financial data APIs, the bot offers personalized financial advice, asset tracking, and educational resources. Key features include a chatbot for financial queries, an asset tracker with performance charts, daily financial news updates, tools for budgeting and expense tracking, and the ability to upload personal financial documents for tailored insights.

---

## Objective

The primary objectives of this project are:

1. **Empower Users with Financial Knowledge**: Provide accessible and personalized financial education to help users make informed decisions.
2. **Personalized Financial Assistance**: Offer customized advice based on individual financial data and goals.
3. **Real-Time Market Insights**: Keep users updated with the latest financial news and asset performance.
4. **Interactive Financial Tools**: Equip users with tools to track expenses, savings, and investments effectively.
5. **Enhanced User Experience**: Create an intuitive and engaging interface that simplifies personal finance management.

---

## Implementation

### Technologies Used

- **Programming Language**: Python
- **Framework**: Streamlit for web app development
- **APIs and Libraries**:
  - **OpenAI API**: For generating conversational responses.
  - **LangChain**: For natural language understanding and processing.
  - **Yahoo Finance (`yfinance`)**: To fetch real-time stock market data.
  - **NewsAPI**: To retrieve the latest financial news headlines.
  - **PyPDF2**: For processing user-uploaded PDF documents.
  - **Chroma Vector Store**: For storing and retrieving processed text data.
  - **CoinGecko API**: To obtain cryptocurrency prices.
- **Data Visualization**: Matplotlib and Streamlit's built-in charting functions.
- **Environment Management**: `dotenv` for handling environment variables securely.

### Key Components

1. **User Interface (UI) Enhancements**:
   - **Sidebar Navigation**: Critical inputs and settings are placed in the sidebar for easy access.
   - **Tab Layout**: The main functionalities are organized into tabs—News, Assets, Chat, and Tools—for better user experience.
   - **Responsive Design**: Ensured compatibility across devices, including mobile.

2. **Financial Data Input**:
   - Users can input their financial details, such as income, expenses, savings, and investment goals.
   - Data is stored in the session state for personalized advice.

3. **Document Upload and Processing**:
   - Users can upload personal financial documents (up to 200MB in PDF format).
   - Uploaded documents are processed using PyPDF2 and stored in a vector database for context-aware responses.

4. **Chatbot Functionality**:
   - Powered by OpenAI's GPT-4 model, the chatbot answers user queries related to personal finance and investments.
   - It utilizes user financial data, uploaded documents, and real-time market information to provide tailored advice.

5. **Asset Tracker and Visualization**:
   - Users can select and track assets of interest.
   - Real-time price data and performance charts are displayed.
   - Price change alerts can be set up based on user-defined thresholds.

6. **Financial News Updates**:
   - Displays the top three trending financial news headlines, updated daily.
   - News data is fetched using the NewsAPI.

7. **Financial Tools**:
   - **Budgeting Tool**: Helps users calculate monthly savings by inputting income and expenses.
   - **Cryptocurrency Prices**: Displays current prices of major cryptocurrencies like Bitcoin and Ethereum.

---

## Impact

The Personal Finance Assistant Bot aims to make financial literacy more accessible by providing:

- **Personalized Advice**: Tailored recommendations help users make rational financial decisions aligned with their goals.
- **Increased Engagement**: Interactive tools and real-time data keep users engaged and informed.
- **Financial Education**: By simplifying complex financial concepts, users can learn and apply knowledge to their personal finances.
- **Empowerment**: Users gain confidence in managing their finances, leading to better financial well-being.

---

## Features

### 1. **Financial Details Input**

- **Purpose**: Allows the bot to provide personalized advice.
- **Functionality**:
  - Users input data such as investment capital, savings, expenses, and financial goals.
  - The data is used to tailor chatbot responses and recommendations.

### 2. **PDF Upload for Personalized Insights**

- **Purpose**: Enables the bot to learn from user-specific documents.
- **Functionality**:
  - Users can upload PDFs (up to 200MB) like bank statements, investment portfolios, or financial plans.
  - The bot processes these documents to extract relevant information.
  - Provides specific advice based on the content of the uploaded documents.

### 3. **Daily Trending Financial News**

- **Purpose**: Keeps users informed about the latest developments in the financial world.
- **Functionality**:
  - Displays the top three financial news headlines updated daily.
  - Headlines are fetched from reputable sources via NewsAPI.
  - Users can click on headlines to read full articles.

### 4. **Asset Tracker with Performance Charts**

- **Purpose**: Helps users monitor the performance of assets they are interested in.
- **Functionality**:
  - Users can select preferred assets from a list or search for specific ones.
  - Real-time data including current price, daily change, and dividend yield is displayed.
  - Interactive charts show historical performance over selected periods.
  - Price alerts can be set for significant changes.

### 5. **Interactive Chatbot**

- **Purpose**: Provides answers to user queries about personal finance and stock investments.
- **Functionality**:
  - Answers are generated using OpenAI's GPT-4 model.
  - The chatbot considers user financial data, uploaded documents, and real-time market information.
  - Supports natural language queries, making interactions intuitive.

### 6. **Financial Tools for Savings and Spendings**

- **Purpose**: Assists users in budgeting and tracking their finances.
- **Functionality**:
  - **Budgeting Tool**: Calculates monthly savings based on income and expenses input.
  - **Expense Tracker**: (Potential Future Feature) Could allow users to categorize and monitor expenses over time.
  - **Savings Goals**: Helps users set and track progress towards financial goals.

---

## Future Improvements

1. **Expansion of Asset Coverage**:
   - **Foreign Exchanges**: Include assets from global markets to cater to international users.
   - **Additional Asset Classes**: Incorporate commodities, bonds, and ETFs.

2. **Cryptocurrency Integration**:
   - **More Cryptocurrencies**: Add a broader range of cryptocurrencies beyond Bitcoin and Ethereum.
   - **Crypto Tools**: Provide tools for tracking crypto portfolios and news specific to the crypto market.

3. **Advanced Data Visualization**:
   - **News Flash Ticker**: Implement a scrolling news ticker for real-time updates.
   - **Interactive Dashboards**: Allow users to customize their dashboards with widgets and charts.

4. **Macro Financial Market Monitoring**:
   - **Economic Indicators**: Display data on interest rates, inflation, GDP growth, etc.
   - **Market Sentiment Analysis**: Provide insights based on market trends and sentiment analysis.

5. **Enhanced Personalization and User Experience**:
   - **User Accounts**: Implement secure user authentication for saving preferences and data.
   - **Notifications and Alerts**: Enable email or push notifications for important updates or alerts.
   - **Multi-language Support**: Extend accessibility by supporting multiple languages.

6. **Educational Content**:
   - **Learning Modules**: Offer tutorials and articles on personal finance topics.
   - **Webinars and Live Sessions**: Host live events with financial experts.

7. **Integration with Financial Services**:
   - **API Connections**: Connect with banks or brokerage accounts for real-time data.
   - **Transaction Analysis**: Automatically categorize and analyze user transactions.

8. **Regulatory Compliance and Security**:
   - **Data Encryption**: Ensure all user data is encrypted and securely stored.
   - **Compliance**: Adhere to financial regulations and data protection laws.

---

## Conclusion

The Personal Finance Assistant Bot serves as a comprehensive platform for individuals seeking to improve their financial literacy and make informed decisions. By combining personalized advice with real-time data and interactive tools, the bot addresses the complexities of personal finance in an accessible manner. The planned future enhancements aim to broaden its capabilities, making it an indispensable tool for users worldwide.

---

*This project underscores the potential of leveraging AI and data analytics to democratize financial education and assistance, ultimately contributing to better financial health and literacy across diverse user groups.*

