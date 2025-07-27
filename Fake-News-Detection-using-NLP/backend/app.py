from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import uvicorn
import numpy as np

from preprocess import preprocess_pipeline

from utils import is_valid_input, log_exception


# Load the trained model and vectorizer
MODEL_PATH = 'model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except Exception as e:
    raise RuntimeError(f"Model or vectorizer loading failed: {e}")

# FastAPI app
app = FastAPI(title="News Classification API")

# Enable CORS for all origins (for development; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("FastAPI server is running and CORS is enabled.")

class NewsInput(BaseModel):
    text: str

@app.post("/predict/")
def predict(news: NewsInput):
    """Predict the class of a news article."""
    try:
        # Input validation!
        if not is_valid_input(news.text):
            raise HTTPException(status_code=400, detail="Input text is empty or invalid.")
        clean_text = preprocess_pipeline(news.text)  # preprocess_pipeline is fine if that’s your process
        vectorized = vectorizer.transform([clean_text])
        prediction = model.predict(vectorized)[0]
        proba = model.predict_proba(vectorized).max()
        # Convert numpy types to Python native types
        if isinstance(prediction, np.generic):
            prediction = prediction.item()
        if isinstance(proba, np.generic):
            proba = proba.item()
        return {
            "prediction": prediction,
            "confidence": float(proba)
        }
    except Exception as e:
        # Log exception for debugging, but return generic error to client
        log_exception("Prediction failed", e)
        raise HTTPException(status_code=500, detail="Server error")


@app.get("/")
def root():
    return {"message": "News Classification API is running."}

# Optional: Uncomment if running directly for local testing
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
