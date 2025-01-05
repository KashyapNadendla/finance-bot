import streamlit as st
import os
from newsapi import NewsApiClient
from newsapi.newsapi_exception import NewsAPIException
from datetime import datetime, timedelta

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def fetch_finance_news(num_articles=3):
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        today = datetime.today().strftime('%Y-%m-%d')
        last_week = (datetime.today() - timedelta(days=7)).strftime('%Y-%m-%d')

        news = newsapi.get_everything(
            q="finance OR economy",
            from_param=last_week,
            to=today,
            language="en",
            sort_by="relevancy",
            page_size=num_articles
        )
        articles = news.get('articles', [])
        return [{"title": article['title'], "url": article['url'], "source": article['source']['name']} for article in articles]
    except NewsAPIException as e:
        if 'rateLimited' in str(e):
            st.warning("News API rate limit exceeded. Please try again later.")
        else:
            st.error("An error occurred while fetching news. Please try again later.")
        return []
    except Exception as e:
        st.error(f"Unexpected error fetching news: {e}")
        return []

def display_finance_news():
    st.subheader("Top Finance News Articles")
    articles = fetch_finance_news(num_articles=3)
    if articles:
        for i, article in enumerate(articles, 1):
            st.markdown(f"[**{i}. {article['title']}**]({article['url']})")
            st.write(f"Source: {article['source']}\n")
    else:
        st.write("No news articles available at this time.")
