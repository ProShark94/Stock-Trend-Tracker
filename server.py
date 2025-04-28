from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
import pandas as pd
import joblib
import os
from datetime import datetime
from utils import combine_financial_news, analyze_sentiments

app = FastAPI()

models_dir = "models/"
data_dir = "data/"
models = {}

async def load_models():
    """Load all models into the application."""
    for symbol in ["AAPL", "MSFT", "NVDA", "AMD", "TSLA"]:
        model_path = os.path.join(models_dir, f"{symbol}_model.joblib")
        if os.path.exists(model_path):
            models[symbol] = joblib.load(model_path)
        else:
            print(f"Model for {symbol} not found at {model_path}")

app.add_event_handler("startup", load_models)

@app.get("/favicon.ico")
async def favicon():
    return Response(content="", media_type="image/x-icon")

@app.get("/predict/{symbol}")
async def predict(symbol: str):
    symbol = symbol.upper()
    if symbol not in models:
        raise HTTPException(status_code=404, detail="Model not found")

    # Financial data path
    financial_data_path = os.path.join(data_dir, f"{symbol}_financial_data.parquet")

    # News data path adjustment based on company name
    company_name = {
        "AAPL": "Apple",
        "MSFT": "Microsoft",
        "NVDA": "Nvidia",
        "AMD": "AMD",
        "TSLA": "Tesla"
    }.get(symbol, symbol)

    news_data_path = os.path.join(data_dir, f"{company_name}_news_data.parquet")

    if not os.path.exists(financial_data_path) or not os.path.exists(news_data_path):
        raise HTTPException(status_code=404, detail="Required data not found")

    financial_data = pd.read_parquet(financial_data_path)
    news_data = pd.read_parquet(news_data_path)
    news_data = analyze_sentiments(news_data)
    
    input_df = combine_financial_news(financial_data, news_data)
    if input_df.empty:
        raise HTTPException(status_code=500, detail="Error in preparing input data")

    prediction = models[symbol].predict(input_df)
    return {"symbol": symbol, "prediction": prediction[0], "date": datetime.now().isoformat()}


@app.get("/")
def read_root():
    return {"message": "Welcome to the Stock Prediction API!"}
