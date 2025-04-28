import uvicorn
from server import app
from server import load_models

@app.on_event("startup")
async def startup_event():
    """Load models and perform any other startup tasks."""
    app.models = load_models()

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
