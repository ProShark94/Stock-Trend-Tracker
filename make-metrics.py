import os
import pandas as pd
import joblib
from utils import (
    read_and_process_financial_data,
    read_and_process_news_data,
    combine_financial_news,
)

data_dir = "data/"
models_dir = "models/"

# List of company symbols for which the models have been trained
companies = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Nvidia": "NVDA",
    "AMD": "AMD",
    "Tesla": "TSLA"
}

for company, symbol in companies.items():
    stock_file = os.path.join(data_dir, f"{symbol}_financial_data.parquet")
    news_file = os.path.join(data_dir, f"{company}_news_data.parquet")
    model_file = os.path.join(models_dir, f"{symbol}_model.joblib")

    print(f"\nEvaluating model for {company}:")
    print("-" * 60)

    if not os.path.exists(stock_file) or not os.path.exists(news_file) or not os.path.exists(model_file):
        print("Missing data/model files for {company}. Please check your directories.")
        continue

    # Read and prepare data
    financial_data = read_and_process_financial_data(stock_file)
    news_sentiments = read_and_process_news_data(news_file)
    combined_data = combine_financial_news(financial_data, news_sentiments)
    combined_df = pd.DataFrame(combined_data)

    # 'market_trend' is the target column
    X_test = combined_df.drop(['market_trend'], axis=1)
    y_test = combined_df['market_trend']

    # Load model and evaluate
    model = joblib.load(model_file)
    try:
        test_score = model.score(X_test, y_test)
        print(f"Test Accuracy: {test_score:.2f}")
        if test_score < 0.95:
            print("!!!!!!!!!!!! WARNING: Model accuracy below 95% !!!!!!!!!!!!")
    except Exception as e:
        print(f"Error evaluating model for {company}: {str(e)}")
