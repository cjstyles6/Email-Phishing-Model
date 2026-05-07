"""Production FastAPI server for phishing email detection."""

import logging
import os
import re
import time
from datetime import date, datetime
from email.parser import Parser
from email.policy import default
from email.utils import parseaddr
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/phishguard_matplotlib")

try:
    import gdown
    import joblib
    import numpy as np
    import shap
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

try:
    from .url_checker import URLChecker
    from .threat_intelligence import ThreatIntelligence
except ImportError:
    from url_checker import URLChecker
    from threat_intelligence import ThreatIntelligence


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "xgboost_model.pkl"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"

VECTORIZER_FILE_ID = "1MfNmLCcmtzGPnoVZRaSGTNo22CFa05j_"
MODEL_FILE_ID = "1gP2BXkwKBqVkJbmsl-gViPc_Y55dKxsY"
VIRUSTOTAL_API_KEY = os.getenv("VIRUSTOTAL_API_KEY")
ABUSEIPDB_API_KEY = os.getenv("ABUSEIPDB_API_KEY")

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
PHISHING_THRESHOLD = 0.75
HEADER_SECURITY_CHECK_COUNT = 8
FREE_EMAIL_PROVIDERS = {"gmail.com", "yahoo.com", "hotmail.com", "outlook.com"}
SUSPICIOUS_X_MAILER_PATTERNS = ("the bat", "massive", "bulk")
DOMAIN_PATTERN = re.compile(
    r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b",
    re.IGNORECASE,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("phishguard_api")


def download_models() -> None:
    """Download model files from Google Drive if they are not present locally."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if not VECTORIZER_PATH.exists():
        logger.info("Vectorizer not found locally. Downloading from Google Drive...")
        gdown.download(
            f"https://drive.google.com/uc?id={VECTORIZER_FILE_ID}",
            str(VECTORIZER_PATH),
            quiet=False,
        )
        logger.info("✅ Vectorizer downloaded successfully")
    else:
        logger.info("✅ Vectorizer found locally")

    if not MODEL_PATH.exists():
        logger.info("Model not found locally. Downloading from Google Drive...")
        gdown.download(
            f"https://drive.google.com/uc?id={MODEL_FILE_ID}",
            str(MODEL_PATH),
            quiet=False,
        )
        logger.info("✅ Model downloaded successfully")
    else:
        logger.info("✅ Model found locally")


class PredictionRequest(BaseModel):
    """Request body for phishing prediction."""
    email_text: str


class BatchPredictionRequest(BaseModel):
    """Request body for batch phishing prediction."""
    emails: list[str]


class GmailHeader(BaseModel):
    """Single Gmail API header item."""
    name: str
    value: str


class HeaderAnalysisRequest(BaseModel):
    """Request body for email header analysis."""
    raw_headers: str | None = None
    gmail_headers: list[GmailHeader] | None = None


class FullAnalysisRequest(HeaderAnalysisRequest):
    """Request body for combined ML and header analysis."""
    email_text: str


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
    probabilities,
    *,
    include_shap: bool = True,
) -> dict[str, Any]:
    """Format model output into the public API response shape."""
    phishing_probability = float(probabilities[1])
    predicted_label = 1 if phishing_probability >= PHISHING_THRESHOLD else 0
    predicted_confidence = float(probabilities[predicted_label]) * 100
    prediction = "Phishing Email" if predicted_label == 1 else "Safe Email"

    result = {
        "prediction": prediction,
        "confidence": round(predicted_confidence, 2),
        "phishing_probability": round(phishing_probability, 4),
        "flags": build_flags(email_text, phishing_probability),
    }
    if include_shap:
        result["shap_explanation"] = get_shap_explanation(email_text)
    return result


def normalize_shap_values(raw_values: Any) -> np.ndarray:
    """Normalize SHAP outputs across binary XGBoost/SHAP variants."""
    if isinstance(raw_values, list):
        raw_values = raw_values[-1]
    values = np.asarray(raw_values)
    if values.ndim == 3:
        values = values[:, :, -1]
    if values.ndim == 2:
        return values[0]
    return values


def normalize_shap_base_value(explainer: Any) -> float:
    """Return a scalar expected value for binary classification explanations."""
    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        return float(expected_value[-1])

    values = np.asarray(expected_value)
    if values.ndim == 0:
        return float(values)
    return float(values.ravel()[-1])


def get_shap_explanation(email_text: str) -> dict[str, Any]:
    """Explain one prediction with top positive and negative SHAP word impacts."""
    email_features = vectorizer.transform([email_text])
    raw_values = shap_explainer.shap_values(email_features)
    shap_values = normalize_shap_values(raw_values)
    feature_values = email_features.toarray()[0]
    feature_names = vectorizer.get_feature_names_out()
    present_indices = np.flatnonzero(feature_values > 0)
    candidate_indices = present_indices if len(present_indices) else np.arange(len(shap_values))

    positive_indices = sorted(
        (index for index in candidate_indices if shap_values[index] > 0),
        key=lambda index: shap_values[index],
        reverse=True,
    )[:10]
    negative_indices = sorted(
        (index for index in candidate_indices if shap_values[index] < 0),
        key=lambda index: shap_values[index],
    )[:10]

    top_phishing_words = [
        {
            "word": str(feature_names[index]),
            "shap_value": round(float(shap_values[index]), 6),
        }
        for index in positive_indices
    ]
    top_safe_words = [
        {
            "word": str(feature_names[index]),
            "shap_value": round(float(shap_values[index]), 6),
        }
        for index in negative_indices
    ]

    phishing_terms = "', '".join(item["word"] for item in top_phishing_words[:3])
    safe_terms = "', '".join(item["word"] for item in top_safe_words[:3])
    if phishing_terms and safe_terms:
        explanation = (
            f"The words '{phishing_terms}' strongly indicated phishing while "
            f"'{safe_terms}' suggested legitimate email."
        )
    elif phishing_terms:
        explanation = f"The words '{phishing_terms}' strongly indicated phishing."
    elif safe_terms:
        explanation = f"The words '{safe_terms}' suggested legitimate email."
    else:
        explanation = "No high-impact words were isolated for this prediction."

    return {
        "top_phishing_words": top_phishing_words,
        "top_safe_words": top_safe_words,
        "base_value": round(normalize_shap_base_value(shap_explainer), 6),
        "prediction_explanation": explanation,
    }


def normalize_domain(domain: str | None) -> str | None:
    """Normalize a domain for safe comparison."""
    if not domain:
        return None
    normalized = domain.lower().strip().strip("<>[]()'\".,;:")
    if normalized.startswith("www."):
        normalized = normalized[4:]
    return normalized or None


def extract_email_domain(header_value: str | None) -> str | None:
    """Extract the domain from an email header value."""
    if not header_value:
        return None

    _, email_address = parseaddr(header_value)
    if "@" not in email_address:
        return None

    return normalize_domain(email_address.rsplit("@", 1)[1])


def extract_display_name_domain(header_value: str | None) -> str | None:
    """Find a domain-like token in the From display name, if one exists."""
    if not header_value:
        return None

    display_name, email_address = parseaddr(header_value)
    actual_domain = extract_email_domain(email_address)
    for match in DOMAIN_PATTERN.findall(display_name):
        candidate = normalize_domain(match)
        if candidate and candidate != actual_domain:
            return candidate
    return None


def add_header_value(
    normalized_headers: dict[str, list[str]],
    header_name: str | None,
    header_value: str | None,
) -> None:
    """Add a header value to the case-insensitive normalized header map."""
    if not header_name:
        return

    name = header_name.strip().lower()
    value = (header_value or "").strip()
    if not name:
        return

    normalized_headers.setdefault(name, []).append(value)


def parse_header_sources(
    raw_headers: str | None,
    gmail_headers: list[GmailHeader] | None,
) -> dict[str, list[str]]:
    """Parse raw or Gmail API headers into a case-insensitive multi-map."""
    normalized_headers: dict[str, list[str]] = {}

    if raw_headers and raw_headers.strip():
        parsed_headers = Parser(policy=default).parsestr(raw_headers)
        for name, value in parsed_headers.items():
            add_header_value(normalized_headers, name, value)

    if gmail_headers:
        for header in gmail_headers:
            add_header_value(normalized_headers, header.name, header.value)

    return normalized_headers


def first_header(
    headers: dict[str, list[str]],
    name: str,
    *,
    contains: str | None = None,
) -> str | None:
    """Return the first matching header value, optionally filtered by text."""
    values = headers.get(name.lower(), [])
    if contains is None:
        return values[0] if values else None

    needle = contains.lower()
    for value in values:
        if needle in value.lower():
            return value
    return None


def extract_header_fields(headers: dict[str, list[str]]) -> dict[str, str | bool | None]:
    """Extract security-relevant email header fields."""
    authentication_results = first_header(
        headers,
        "authentication-results",
        contains="dmarc",
    )

    return {
        "from_address": first_header(headers, "from"),
        "reply_to": first_header(headers, "reply-to"),
        "return_path": first_header(headers, "return-path"),
        "received_spf": first_header(headers, "received-spf"),
        "dkim_signature": bool(headers.get("dkim-signature")),
        "dmarc": authentication_results
        or first_header(headers, "dmarc-filter")
        or first_header(headers, "x-dmarc-status"),
        "x_mailer": first_header(headers, "x-mailer"),
        "message_id": first_header(headers, "message-id"),
        "subject": first_header(headers, "subject"),
        "date": first_header(headers, "date"),
    }


def build_header_flags(extracted_headers: dict[str, str | bool | None]) -> list[str]:
    """Run phishing-oriented security checks against extracted headers."""
    flags: list[str] = []
    from_value = extracted_headers["from_address"]
    from_domain = extract_email_domain(from_value if isinstance(from_value, str) else None)

    reply_to_domain = extract_email_domain(
        extracted_headers["reply_to"]
        if isinstance(extracted_headers["reply_to"], str)
        else None
    )
    if from_domain and reply_to_domain and reply_to_domain != from_domain:
        flags.append("reply_to_mismatch")

    return_path_domain = extract_email_domain(
        extracted_headers["return_path"]
        if isinstance(extracted_headers["return_path"], str)
        else None
    )
    if from_domain and return_path_domain and return_path_domain != from_domain:
        flags.append("return_path_mismatch")

    received_spf = extracted_headers["received_spf"]
    if isinstance(received_spf, str) and (
        "softfail" in received_spf.lower() or "fail" in received_spf.lower()
    ):
        flags.append("spf_fail")

    if not extracted_headers["dkim_signature"]:
        flags.append("no_dkim")

    if from_domain in FREE_EMAIL_PROVIDERS:
        flags.append("free_email_provider")

    x_mailer = extracted_headers["x_mailer"]
    if isinstance(x_mailer, str) and any(
        pattern in x_mailer.lower() for pattern in SUSPICIOUS_X_MAILER_PATTERNS
    ):
        flags.append("suspicious_x_mailer")

    if not extracted_headers["message_id"]:
        flags.append("no_message_id")

    display_domain = extract_display_name_domain(
        from_value if isinstance(from_value, str) else None
    )
    if from_domain and display_domain and display_domain != from_domain:
        flags.append("domain_mismatch")

    return flags


def threat_level_from_score(score: float) -> str:
    """Map a normalized threat score to a user-facing risk level."""
    if score < 0.3:
        return "safe"
    if score < 0.6:
        return "suspicious"
    return "danger"


def build_header_recommendation(flags: list[str], threat_level: str) -> str:
    """Create a concise English recommendation for the header analysis result."""
    if not flags:
        return (
            "No major header anomalies were detected. Continue to review links, "
            "attachments, and message content before trusting the email."
        )

    if threat_level == "danger":
        return (
            "Multiple header authentication or identity anomalies were detected. "
            "Treat this email as high risk, avoid clicking links or opening attachments, "
            "and verify the sender through a trusted channel."
        )

    if "spf_fail" in flags or "no_dkim" in flags:
        return (
            "The message shows authentication weaknesses. Verify the sender before "
            "responding, opening attachments, or following any requested action."
        )

    return (
        "Some sender identity signals look unusual. Use caution and confirm the "
        "message through an independent trusted contact path."
    )


def analyze_header_payload(
    raw_headers: str | None,
    gmail_headers: list[GmailHeader] | None,
) -> dict[str, Any]:
    """Analyze email headers and return extracted fields, flags, and risk score."""
    has_raw_headers = bool(raw_headers and raw_headers.strip())
    has_gmail_headers = bool(gmail_headers)
    if not has_raw_headers and not has_gmail_headers:
        raise HTTPException(
            status_code=400,
            detail="raw_headers or gmail_headers must be provided",
        )

    headers = parse_header_sources(raw_headers, gmail_headers)
    extracted_headers = extract_header_fields(headers)
    flags = build_header_flags(extracted_headers)
    header_threat_score = round(len(flags) / HEADER_SECURITY_CHECK_COUNT, 4)
    threat_level = threat_level_from_score(header_threat_score)

    return {
        "extracted_headers": extracted_headers,
        "flags": flags,
        "header_threat_score": header_threat_score,
        "threat_level": threat_level,
        "recommendation": build_header_recommendation(flags, threat_level),
    }


def predict_email_text(email_text: str, *, include_shap: bool = True) -> dict[str, Any]:
    """Run the ML phishing model for a single email body."""
    email_features = vectorizer.transform([email_text])
    probabilities = model.predict_proba(email_features)[0]
    return format_prediction_result(email_text, probabilities, include_shap=include_shap)


def build_full_analysis_recommendation(
    ml_analysis: dict[str, Any],
    header_analysis: dict[str, Any],
    url_analysis: dict[str, Any],
    threat_intelligence: dict[str, Any],
    final_verdict: str,
) -> str:
    """Summarize the combined ML and header-analysis result."""
    if final_verdict == "danger":
        return (
            "The combined content, header, URL, and threat intelligence signals indicate high phishing risk. "
            "Do not interact with the email until it has been verified independently."
        )

    if final_verdict == "suspicious":
        return (
            "The email has enough suspicious indicators to warrant caution. Review "
            "the ML, header, URL, and threat intelligence findings, then verify the sender before acting."
        )

    if (
        ml_analysis.get("flags")
        or header_analysis.get("flags")
        or url_analysis.get("suspicious_count", 0)
        or url_analysis.get("shortener_count", 0)
        or threat_intelligence.get("flags")
    ):
        return (
            "The overall score is low, but one or more cautionary signals were found. "
            "Review the highlighted flags before trusting the message."
        )

    return "The email appears low risk based on both content and header checks."


# Download models before anything else runs
download_models()

app = FastAPI(title="Email Phishing Detection API")
model, vectorizer = load_artifacts()
shap_explainer = shap.TreeExplainer(model)
url_checker = URLChecker(api_key=VIRUSTOTAL_API_KEY)
threat_intelligence = ThreatIntelligence(abuseipdb_api_key=ABUSEIPDB_API_KEY)

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
def model_info() -> dict[str, float | int | str | bool]:
    """Return metadata for the currently deployed phishing model."""
    return {
        **MODEL_INFO,
        "classification_threshold": PHISHING_THRESHOLD,
        "last_updated": date.today().isoformat(),
        "virustotal_enabled": bool(VIRUSTOTAL_API_KEY),
        "abuseipdb_enabled": bool(ABUSEIPDB_API_KEY),
        "whois_enabled": True,
    }


@app.post("/predict")
def predict_email(request: PredictionRequest) -> dict[str, Any]:
    """Predict whether an email is phishing and return supporting metadata."""
    return predict_email_text(request.email_text)


@app.post("/explain")
def explain_prediction(request: PredictionRequest) -> dict[str, Any]:
    """Return only SHAP explanation details for a single email body."""
    return get_shap_explanation(request.email_text)


@app.post("/analyze-headers")
def analyze_headers(request: HeaderAnalysisRequest) -> dict[str, Any]:
    """Analyze email authentication and routing headers for phishing signals."""
    return analyze_header_payload(request.raw_headers, request.gmail_headers)


@app.post("/check-urls")
def check_urls(request: PredictionRequest) -> dict[str, Any]:
    """Extract URLs from email text and check local and reputation signals."""
    return url_checker.check_email_text(request.email_text)


@app.post("/threat-intelligence")
async def analyze_threat_intelligence(request: HeaderAnalysisRequest) -> dict[str, Any]:
    """Analyze sender IP reputation and sender domain age intelligence."""
    has_raw_headers = bool(request.raw_headers and request.raw_headers.strip())
    has_gmail_headers = bool(request.gmail_headers)
    if not has_raw_headers and not has_gmail_headers:
        raise HTTPException(
            status_code=400,
            detail="raw_headers or gmail_headers must be provided",
        )

    return await threat_intelligence.analyze(
        raw_headers=request.raw_headers,
        gmail_headers=request.gmail_headers,
    )


@app.post("/full-analysis")
async def full_analysis(request: FullAnalysisRequest) -> dict[str, Any]:
    """Run ML content classification, header forensics, and URL reputation checks."""
    ml_analysis = predict_email_text(request.email_text)
    header_analysis = analyze_header_payload(request.raw_headers, request.gmail_headers)
    url_analysis = url_checker.check_email_text(request.email_text)
    threat_intel_analysis = await threat_intelligence.analyze(
        raw_headers=request.raw_headers,
        gmail_headers=request.gmail_headers,
    )

    ml_score = float(ml_analysis["phishing_probability"])
    header_score = float(header_analysis["header_threat_score"])
    url_score = float(url_analysis["url_threat_score"])
    threat_intel_score = float(threat_intel_analysis["threat_intelligence_score"])
    combined_threat_score = round(
        (ml_score * 0.5)
        + (header_score * 0.15)
        + (url_score * 0.15)
        + (threat_intel_score * 0.2),
        4,
    )
    final_verdict = threat_level_from_score(combined_threat_score)

    return {
        "ml_analysis": ml_analysis,
        "header_analysis": header_analysis,
        "url_analysis": url_analysis,
        "threat_intelligence": threat_intel_analysis,
        "combined_threat_score": combined_threat_score,
        "final_verdict": final_verdict,
        "recommendation": build_full_analysis_recommendation(
            ml_analysis,
            header_analysis,
            url_analysis,
            threat_intel_analysis,
            final_verdict,
        ),
    }


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
    probabilities = model.predict_proba(email_features)
    processing_time_ms = (time.perf_counter() - started_at) * 1000

    results = [
        {
            "email_index": index,
            **format_prediction_result(
                email_text,
                email_probabilities,
                include_shap=False,
            ),
        }
        for index, (email_text, email_probabilities) in enumerate(
            zip(request.emails, probabilities)
        )
    ]

    return {
        "results": results,
        "processing_time_ms": round(processing_time_ms, 2),
    }


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
