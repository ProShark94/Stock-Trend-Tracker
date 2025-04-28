import requests
import os
import pandas as pd
from utils import process_financial_data, process_news_data


# Ensure data directory exists
data_directory = "data"
if not os.path.exists(data_directory):
    os.makedirs(data_directory)
# AlphaVantage
def fetch_financial_data(symbol):
    ALPHA_VANTAGE_API_KEY = 'KSR0XHEWLKH5EE5X'
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={ALPHA_VANTAGE_API_KEY}"
    response = requests.get(url)
    data = response.json()

    if "Time Series (5min)" not in data:
        raise ValueError("API call failed or returned unexpected data structure:", data.get("Note", data.get("Information", "")))
    df = process_financial_data(data)
    file_path = f"data/{symbol}_financial_data.parquet"
    df.to_parquet(file_path)
    print(f"File saved at: {file_path}")
    return df

# NewsAPI
def fetch_news_data(query):
    NEWS_API_KEY = 'ce339a03ac454c3c82bf3bf656a11079'
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    data = response.json()

    news_items = process_news_data(data)
    df = pd.DataFrame(news_items)
    file_path = f"data/{query}_news_data.parquet"
    df.to_parquet(file_path)
    print(f"File saved at: {file_path}")
    return df

def fetch_and_store_all_data():
    companies = {
        "Apple": "AAPL",
        "Microsoft": "MSFT",
        "Nvidia": "NVDA",
        "AMD": "AMD",
        "Tesla": "TSLA"
    }

    for company, symbol in companies.items():
        print(f"Fetching and storing data for {company}")
        fetch_financial_data(symbol)
        fetch_news_data(company)

if __name__ == "__main__":
    fetch_and_store_all_data()
