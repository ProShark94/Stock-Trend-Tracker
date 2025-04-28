import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def process_financial_data(data):
    # Extract the time series data
    time_series_data = data.get('Time Series (5min)', {})
    
    processed_data = {time: {
        "open": float(details["1. open"]),
        "high": float(details["2. high"]),
        "low": float(details["3. low"]),
        "close": float(details["4. close"]),
        "volume": int(details["5. volume"])
    } for time, details in time_series_data.items()}
    
    return pd.DataFrame.from_dict(processed_data, orient='index')


# Function to process news data and extract headlines and descriptions
def process_news_data(data):
    articles = data.get('articles', [])
    return [{'title': article['title'], 'description': article['description']} for article in articles]


def analyze_sentiments(news_df):
    analyzer = SentimentIntensityAnalyzer()
    news_df['title'] = news_df['title'].astype(str)
    news_df['description'] = news_df['description'].astype(str)
    news_df['text'] = news_df['title'] + " " + news_df['description']
    news_df['sentiment'] = news_df['text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

    return news_df

def calculate_financial_trend(financial_df):
    """Calculate financial trend based on recent price movements."""
    # Simple moving averages (SMA) for the short and long term
    short_window = 5
    long_window = 20
    financial_df['short_mavg'] = financial_df['close'].rolling(window=short_window, min_periods=1).mean()
    financial_df['long_mavg'] = financial_df['close'].rolling(window=long_window, min_periods=1).mean()

    # Trend determination
    if financial_df['short_mavg'].iloc[-1] > financial_df['long_mavg'].iloc[-1]:
        return "Uptrend", financial_df['short_mavg'].iloc[-1] - financial_df['long_mavg'].iloc[-1]
    else:
        return "Downtrend", financial_df['short_mavg'].iloc[-1] - financial_df['long_mavg'].iloc[-1]

def calculate_sentiment_strength(news_df):
    """Calculate the average sentiment and its strength from a DataFrame."""
    if 'sentiment' not in news_df.columns:
        raise ValueError("DataFrame must contain 'sentiment' column")

    average_sentiment = news_df['sentiment'].mean()
    sentiment_strength = news_df['sentiment'].std()

    return average_sentiment, sentiment_strength

def combine_financial_news(financial_df, news_sentiments):

    financial_trend, trend_strength = calculate_financial_trend(financial_df)
    average_sentiment, sentiment_strength = calculate_sentiment_strength(news_sentiments)
    overall_sentiment = "Positive" if average_sentiment > 0 else "Negative"

    if financial_trend == "Uptrend" and overall_sentiment == "Positive":
        prediction = "Strong Bullish"
    elif financial_trend == "Downtrend" and overall_sentiment == "Negative":
        prediction = "Strong Bearish"
    else:
        if abs(trend_strength) > abs(average_sentiment):
            prediction = "Mild " + financial_trend
        else:
            prediction = "Mildly " + overall_sentiment

    
    financial_df['market_trend'] = prediction  # Assigning prediction to a new column

    return financial_df  


# Function to save processed data to Parquet file. Function will be used only if reading fails.
def save_processed_data(data, filename):
    if isinstance(data, pd.DataFrame):
        data.to_parquet(filename)
    else:
        raise ValueError("Expected a Pandas DataFrame")

# Read and process financial data from a Parquet file
def read_and_process_financial_data(filename):
    df = pd.read_parquet(filename)
    return df

# Read and process news data from a Parquet file
def read_and_process_news_data(filename):
    df = pd.read_parquet(filename)
    sentiments = analyze_sentiments(df)
    return sentiments

def prepare_data_ml(stock_file, news_file):
    financial_data = read_and_process_financial_data(stock_file)
    news_sentiments = read_and_process_news_data(news_file)
    
    # Combine financial data and news sentiments
    combined_data = combine_financial_news(financial_data, news_sentiments)
    
    X = combined_data.drop("market_trend", axis=1)
    y = combined_data["market_trend"]
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, y_train, X_test, y_test
   
def count_missing_values(df):
    """Returns the count of missing values in each column of the DataFrame."""
    return df.isnull().sum()

def check_data():
    data_dir = "data/"
    for filename in os.listdir(data_dir):
        if filename.endswith(".parquet"):
            print("\n\n" + filename, "-" * 60 + "\n")
            file_path = os.path.join(data_dir, filename)
            # Read the data from the Parquet file
            df = pd.read_parquet(file_path)
            # Print the count of missing values
            missing_values = count_missing_values(df)
            print("Missing Values Count:\n", missing_values)
            # Print the DataFrame's content
            print("\nData Preview:\n", df.head())  
            print("\n" + "-" * 60 + "\n")
            
            
# Example function to demonstrate usage
def analyze_stock(stock_file, news_file):
    financial_data = read_and_process_financial_data(stock_file)
    news_sentiments = read_and_process_news_data(news_file)
    return combine_financial_news(financial_data, news_sentiments)

