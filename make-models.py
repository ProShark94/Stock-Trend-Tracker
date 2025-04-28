from fileinput import filename
import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from utils import (
    read_and_process_financial_data, read_and_process_news_data, prepare_data_ml,combine_financial_news)


def train_model(X_train, y_train):
    numeric_features = ["open", "high", "low", "close", "volume"]
    numeric_transformer = Pipeline(
        steps=[
            ("Median Imputer", SimpleImputer(strategy="median")),
            ("Standardization", StandardScaler()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[("Numeric Transformer", numeric_transformer, numeric_features)],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("Preprocessor", preprocessor),
            ("Classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )

    param_grid = {
        "Classifier__max_depth": [5, 10, 15],
        "Classifier__min_samples_split": [2, 5],
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy")
    grid_search.fit(X_train, y_train)
    return grid_search

data_dir = "data/"
models_dir = "models/"

# check models directory exists
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

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

    if not os.path.exists(stock_file) or not os.path.exists(news_file):
        print(f"Data files missing for {company}. Please check your data directory.")
        continue

    print(f"\nTraining model for {company}:")
    
    financial_data = read_and_process_financial_data(stock_file)
    news_sentiments = read_and_process_news_data(news_file)
    combined_data = combine_financial_news(financial_data, news_sentiments)
    print("Columns in combined data:", combined_data.columns.tolist())

    X_train, y_train, X_test, y_test = prepare_data_ml(stock_file, news_file)

    # Train the model using the training data
    model = train_model(X_train, y_train)
    joblib.dump(model, os.path.join(models_dir, f"{symbol}_model.joblib"))
    print(f"Model saved for {company}")

    # Evaluate the model using the test data
    test_accuracy = model.score(X_test, y_test)
    print(f"Test Accuracy for {company}: {test_accuracy:.2f}")

    # Print the best score
    if hasattr(model, 'best_score_'):
        print(f"CV Accuracy for {company}: {model.best_score_:.2f}")
