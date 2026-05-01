"""Production FastAPI server for phishing email detection."""

import logging
import os
import time
from datetime import date, datetime
from pathlib import Path

try:
    import joblib
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing API dependency.\n"
        "Install the project requirements first with:\n"
        "  pip install -r requirements.txt"
    ) from exc


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "xgboost_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"

MODEL_INFO = {
    "model_version": "2.0",
    "training_samples": 179052,
    "accuracy": 0.9801,
    "precision": 0.9700,
    "recall": 0.9918,
    "f1_score": 0.9808,
}

URGENT_LANGUAGE_TRIGGERS = (
    "verify",
    "suspend",
    "click here",
    "urgent",
    "account",
    "password",
    "bank",
)
SENDER_SPOOFING_TRIGGERS = ("dear customer", "dear user", "dear account holder")
CREDENTIAL_HARVESTING_TRIGGERS = (
    "username",
    "password",
    "login",
    "sign in",
    "credentials",
)
FINANCIAL_LURE_TRIGGERS = (
    "winner",
    "prize",
    "lottery",
    "million dollars",
    "inheritance",
    "next of kin",
)
IMPERSONATION_TRIGGERS = (
    "paypal",
    "amazon",
    "apple",
    "microsoft",
    "netflix",
    "bank of",
)
MAX_BATCH_SIZE = 50

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("phishguard_api")


class PredictionRequest(BaseModel):
    """Request body for phishing prediction."""

    email_text: str


class BatchPredictionRequest(BaseModel):
    """Request body for batch phishing prediction."""

    emails: list[str]


def load_artifacts() -> tuple:
    """Load the trained model and vectorizer once when the app starts."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Place 'xgboost_model.pkl' inside the models directory."
        )

    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            f"TF-IDF vectorizer not found: {VECTORIZER_PATH}\n"
            "Place 'tfidf_vectorizer.pkl' inside the models directory."
        )

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def build_flags(email_text: str, phishing_probability: float) -> list[str]:
    """Create simple explanatory flags based on score and message content."""
    flags: list[str] = []
    normalized_text = email_text.lower()

    if phishing_probability > 0.9:
        flags.append("high confidence phishing detected")

    if any(keyword in normalized_text for keyword in URGENT_LANGUAGE_TRIGGERS):
        flags.append("urgent/sensitive language detected")

    if "http" in normalized_text or "www" in normalized_text:
        flags.append("contains URLs")

    if any(keyword in normalized_text for keyword in SENDER_SPOOFING_TRIGGERS):
        flags.append("sender spoofing language")

    if any(keyword in normalized_text for keyword in CREDENTIAL_HARVESTING_TRIGGERS):
        flags.append("credential harvesting")

    if any(keyword in normalized_text for keyword in FINANCIAL_LURE_TRIGGERS):
        flags.append("financial lure")

    if any(keyword in normalized_text for keyword in IMPERSONATION_TRIGGERS):
        flags.append("impersonation attempt")

    return flags


def format_prediction_result(
    email_text: str,
    predicted_label: int,
    probabilities,
) -> dict[str, float | str | list[str]]:
    """Format model output into the public API response shape."""
    phishing_probability = float(probabilities[1])
    predicted_confidence = float(probabilities[predicted_label]) * 100
    prediction = "Phishing Email" if predicted_label == 1 else "Safe Email"

    return {
        "prediction": prediction,
        "confidence": round(predicted_confidence, 2),
        "phishing_probability": round(phishing_probability, 4),
        "flags": build_flags(email_text, phishing_probability),
    }


app = FastAPI(title="Email Phishing Detection API")
model, vectorizer = load_artifacts()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with timing metadata for lightweight observability."""
    started_at = time.perf_counter()
    timestamp = datetime.now().isoformat(timespec="seconds")

    try:
        return await call_next(request)
    finally:
        response_time_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "%s %s %s %.2fms",
            timestamp,
            request.method,
            request.url.path,
            response_time_ms,
        )


@app.get("/health")
def health_check() -> dict[str, str]:
    """Return a basic server health status."""
    return {"status": "ok"}


@app.get("/model-info")
def model_info() -> dict[str, float | int | str]:
    """Return metadata for the currently deployed phishing model."""
    return {
        **MODEL_INFO,
        "last_updated": date.today().isoformat(),
    }


@app.post("/predict")
def predict_email(request: PredictionRequest) -> dict[str, float | str | list[str]]:
    """Predict whether an email is phishing and return supporting metadata."""
    email_features = vectorizer.transform([request.email_text])

    predicted_label = int(model.predict(email_features)[0])
    probabilities = model.predict_proba(email_features)[0]

    return format_prediction_result(request.email_text, predicted_label, probabilities)


@app.post("/predict-batch")
def predict_batch(
    request: BatchPredictionRequest,
) -> dict[str, float | list[dict[str, float | int | str | list[str]]]]:
    """Predict phishing risk for up to 50 emails in a single model batch."""
    email_count = len(request.emails)

    if email_count == 0:
        raise HTTPException(status_code=400, detail="emails must not be empty")

    if email_count > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"emails must contain no more than {MAX_BATCH_SIZE} items",
        )

    started_at = time.perf_counter()
    email_features = vectorizer.transform(request.emails)
    predicted_labels = model.predict(email_features)
    probabilities = model.predict_proba(email_features)
    processing_time_ms = (time.perf_counter() - started_at) * 1000

    results = [
        {
            "email_index": index,
            **format_prediction_result(
                email_text,
                int(predicted_label),
                email_probabilities,
            ),
        }
        for index, (email_text, predicted_label, email_probabilities) in enumerate(
            zip(request.emails, predicted_labels, probabilities)
        )
    ]

    return {
        "results": results,
        "processing_time_ms": round(processing_time_ms, 2),
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
