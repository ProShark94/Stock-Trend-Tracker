# CS 498 E2E Final Project (sp24) repo for NetID: dr31

GitHub username at initialization time: ProShark94

Final Project CS 498: Stock Predictor using Sentimental Analysis 

This Project follows similar path as MP03 but with a practical touch on markets and emphasising the trends in markets.
In this we will use top 5 stocks in the market pertaining to Technology and Manufacturing.
data and model folders are similar to MP03.

This is a simple and efficient predictor based on solving a simple problem correctly is better than tackling a huge problem and producing an inoperable product. 
_____________________________________________________________________________________________________________________________________________________

# Stock Trend Predictor API

The purpose of this product is to predict market trends based on a combination of financial data trends and news sentiment analysis. This solution provides insights into whether a stock is likely to experience an uptrend or downtrend, which can be crucial for traders and investors making buying or selling decisions. 

These Stocks are:

<b>
"AAPL": "Apple Inc.",


"MSFT": "Microsoft Corporation",

"NVDA": "NVIDIA Corporation",

"AMD": "Advanced Micro Devices",

"TSLA": "Tesla Inc"
<b>


The 5 Stocks can be changed in make-data.py in the function fetch_and_store_all_data. Note that the make-data.py and make-models.py have to be run again to generate the models. Ensure the mapping in utils, server and client is updated. Do this only if you would like to check for another stock.

**Context**

In the volatile world of stock trading, timely and accurate predictions can be the key to success. This API leverages machine learning algorithm like Random Forest classifier and sentiment analysis to interpret current market data and news trends, offering predictions on future trends.


**API USED**

We are using two API's to get data.
- Alpha Vantage API: https://www.alphavantage.co/documentation/
- News API : https://newsapi.org

The API keys are already loaded in make-data.py. Please note that there is restriction of 25 API calls per day. if you wish to update the data files please run it post market closure as the data takes the 20+ years historical data plus the last traded day for Alpha vantage and News API is genrally articles less than a week old. A better news data or API with current articles can do amuch better job. As those API are not free, hence we have to make do with what is open source for now.


## Key Functionalities:
1. Data Processing and Extraction:
    - process_financial_data(data): Extracts and processes time-series financial data from a given JSON-like structure, organizing it into a structured DataFrame.
    - process_news_data(data): Extracts headlines and descriptions from news data to prepare for sentiment analysis.
2. Sentiment Analysis:
    - analyze_sentiments(news_df): Enhances the news DataFrame by calculating sentiment scores for each article using the VADER sentiment analysis tool. This function adds a new column for sentiment scores, combining the article's title and description to get a comprehensive sentiment overview.
3. Financial Trend Calculation:
    - calculate_financial_trend(financial_df): Determines the financial trend of a stock by calculating short-term and long-term moving averages of closing prices. The function assesses whether the stock is in an uptrend or downtrend based on these averages.
4. Sentiment Strength Calculation:
    - calculate_sentiment_strength(news_df): Calculates the average sentiment and its standard deviation (as a measure of sentiment strength) from the sentiment scores added to the news DataFrame.
5. Combining Financial and Sentiment Data:
    - combine_financial_news(financial_df, news_sentiments): Integrates financial trends with news sentiment data to make a holistic market trend prediction. This function outputs a DataFrame with the predicted market trend based on both financial movements and news sentiment.
6. Data Preparation:
    - prepare_data_ml(stock_file, news_file): Facilitates the creation of training and testing datasets by reading, processing, and combining financial and news data. This function returns training and testing splits, ready for use in machine learning models.
7. Data Management and Utility Functions:
    - save_processed_data(data, filename): Saves a given DataFrame to a Parquet file. This is useful for caching or saving transformed data.
    - read_and_process_financial_data(filename): Reads financial data from a Parquet file and returns a DataFrame.
    - read_and_process_news_data(filename): Reads news data from a Parquet file, applies sentiment analysis, and returns the processed DataFrame.
    - count_missing_values(df): Calculates the count of missing values in each column of a DataFrame, aiding in data quality checks.
    - check_data(): Provides a utility for checking and displaying data from all Parquet files in a specified directory, including previews and missing value counts.


## Setup Instructions

- Requirements
    - python 3.8
    - uv(can use pip as well)
    - Virtual environment(Recommended)

- Installation

1. Clone the Repository:
    - git clone https://github.com/illinois-cs-coursework/sp24_cs498e2e-final_dr31.git

2. Setup Virtual environment:
    - use uv(Recommended- its fast , Not kidding!) : https://github.com/astral-sh/uv
    - can also use default virtual environments

3. Install Dependencies
    - uv pip install -r requirements.txt or  pip install -r requirements.txt

4. Run Files:
    - 'run.py' file runs and start the relevant server 
    - With the server running, run 'client.py' which should give the predictive trend for the stocks
    - Optional: run 'make-metrics.py' to calculate metrics for the 5 stocks.
    


## Example Output

The output of our prediction tool is determined by analyzing both the financial trends and the sentiment of related news articles. Here's how our predictions might read under different circumstances:

1. Strong Bullish: When both the financial trend indicates an uptrend (prices rising) and news sentiment is predominantly positive, the prediction will be "Strong Bullish." This suggests that the market conditions are very favorable, and it might be an excellent time to consider buying or holding the stock.
    - Example: If Apple's stock prices have been rising over the past period and recent news articles are optimistic about Apple’s market performance and product innovations, the output would likely be "Strong Bullish."
2. Strong Bearish: Conversely, if the financial trend is a downtrend (prices falling) and news sentiment is overwhelmingly negative, the prediction will be "Strong Bearish." This indicates unfavorable market conditions, potentially advising to sell or avoid buying the stock.
    - Example: If Tesla's stock prices have been consistently falling, coupled with negative press regarding production delays or legal issues, the predicted trend would be "Strong Bearish."

3. Mild Trends: In cases where there is a mix of signals or less pronounced trends:
    - If the financial data shows an uptrend but with less conviction (i.e., a smaller gap between short and long-term moving averages) or if the sentiment analysis is less distinctly positive, the output may be "Mild Uptrend."
    - Similarly, a "Mild Downtrend" might be predicted if the stock is experiencing a slight downtrend or mixed news sentiment that doesn't strongly support a bearish outlook.
    - The term "Mildly Positive" or "Mildly Negative" could be used if the sentiment analysis results are more dominant in the decision process but are not overwhelmingly strong.
        - Example: Suppose Microsoft’s stock shows a slight increase in price but mixed news sentiment. If the trend strength (based on moving averages) is not significantly stronger than the sentiment, the prediction might be "Mild Uptrend." If sentiment is more distinctly negative but the price downtrend is mild, the prediction might be "Mildly Negative."

These predictions are designed to help users make more informed decisions by considering both numerical financial data and qualitative insights from news sentiment. This dual-analysis approach allows for a nuanced understanding of potential market movements, catering to both conservative and aggressive investment strategies.

### Development File
- A scratch file called scratch.ipynb is used for testing purposes and can be referred to if you want to modify the API functions for your own usage or modifications. This is not needed for the package and can be deleted if not used.