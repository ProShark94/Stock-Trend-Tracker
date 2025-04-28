import requests

def get_predictions(symbols):
    predictions = {}
    url_template = "http://127.0.0.1:8000/predict/{}"

    # Mapping symbols to full names. Just improves the readability
    full_names = {
        "AAPL": "Apple Inc.",
        "MSFT": "Microsoft Corporation",
        "NVDA": "NVIDIA Corporation",
        "AMD": "Advanced Micro Devices",
        "TSLA": "Tesla Inc"
    }

    for symbol in symbols:
        try:
            response = requests.get(url_template.format(symbol))
            response.raise_for_status()  # This will raise an exception for HTTP errors
            # Add the full name to the response JSON for easier reference
            prediction_data = response.json()
            prediction_data['full_name'] = full_names.get(symbol, "Unknown")
            predictions[symbol] = prediction_data
        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error occurred for {symbol} ({full_names.get(symbol, 'Unknown')}): {http_err} - Response Content: {response.text}")
            predictions[symbol] = None
        except Exception as err:
            print(f"An error occurred for {symbol} ({full_names.get(symbol, 'Unknown')}): {err}")
            predictions[symbol] = None

    return predictions

# Example usage for multiple stocks
symbols = ["AAPL", "MSFT", "NVDA", "AMD", "TSLA"]
all_predictions = get_predictions(symbols)

for symbol, prediction in all_predictions.items():
    if prediction:
        print(f"Prediction for {prediction['full_name']} ({symbol}): {prediction['prediction']}")
    else:
        print(f"Failed to get prediction for {prediction['full_name']} ({symbol}).")
