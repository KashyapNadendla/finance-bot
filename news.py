# # news.py
# import streamlit as st
# from datetime import datetime, timedelta
# from newsapi import NewsApiClient

# @st.cache_data(ttl=86400)
# def fetch_finance_news():
#     """
#     Fetch the top finance news articles from the past week using NewsAPI.
#     Cached for 24 hours (86400 seconds) to avoid excessive API calls.
#     """
#     # NOTE: If you store your NEWS_API_KEY in session_state, fetch it from there.
#     # Otherwise, you can use an environment variable directly:
#     # from dotenv import load_dotenv
#     # load_dotenv()
#     # import os
#     # NEWS_API_KEY = os.getenv("NEWS_API_KEY")
#     NEWS_API_KEY = st.session_state.get('NEWS_API_KEY', None)
#     if not NEWS_API_KEY:
#         st.error("NEWS_API_KEY not set. Cannot fetch news.")
#         return []

#     newsapi = NewsApiClient(api_key=NEWS_API_KEY)
#     try:
#         today = datetime.today().strftime('%Y-%m-%d')
#         last_week = (datetime.today() - timedelta(days=7)).strftime('%Y-%m-%d')
#         news = newsapi.get_everything(
#             q="finance OR economy",
#             from_param=last_week,
#             to=today,
#             language="en",
#             sort_by="relevancy",
#             page_size=3
#         )
#         articles = news.get('articles', [])
#         return [
#             {
#                 "title": article['title'],
#                 "url": article['url'],
#                 "source": article['source']['name']
#             }
#             for article in articles
#         ]
#     except Exception as e:
#         if 'rateLimited' in str(e):
#             st.warning("News API rate limit exceeded. Please try again later.")
#         else:
#             st.error(f"An error occurred while fetching news: {e}")
#         return []

# def display_finance_news():
#     """
#     Display the top finance news articles using Streamlit.
#     """
#     st.subheader("Top 3 Finance News Articles Today")
#     articles = fetch_finance_news()
#     if articles:
#         for i, article in enumerate(articles, 1):
#             st.markdown(f"[**{i}. {article['title']}**]({article['url']})")
#             st.write(f"Source: {article['source']}")
#     else:
#         st.write("No news articles available at this time.")

# news.py
import streamlit as st
import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
load_dotenv()

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

@st.cache_data(ttl=86400)
def fetch_finance_news():
    """
    Fetch the top finance news articles using Alpha Vantage's NEWS_SENTIMENT endpoint.
    Cached for 24 hours.
    """
    if not ALPHA_VANTAGE_API_KEY:
        st.error("ALPHA_VANTAGE_API_KEY not set. Cannot fetch news.")
        return []
    
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&apikey={ALPHA_VANTAGE_API_KEY}"
    try:
        response = requests.get(url)
        data = response.json()
        feed = data.get("feed", [])
        # Extract the top 3 articles (or fewer if not enough available)
        articles = []
        for article in feed[:3]:
            articles.append({
                "title": article.get("title", "No Title"),
                "url": article.get("url", "#"),
                "source": article.get("source", "Unknown")
            })
        return articles
    except Exception as e:
        st.error(f"An error occurred while fetching news: {e}")
        return []

def display_finance_news():
    st.subheader("Top 3 Finance News Articles Today")
    articles = fetch_finance_news()
    if articles:
        for i, article in enumerate(articles, 1):
            st.markdown(f"[**{i}. {article['title']}**]({article['url']})")
            st.write(f"Source: {article['source']}")
    else:
        st.write("No news articles available at this time.")
