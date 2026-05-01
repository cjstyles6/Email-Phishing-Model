"""Smoke-test the PhishGuard prediction API with sample emails."""

import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_URL = os.getenv("PHISHGUARD_API_URL", "http://localhost:8000/predict")

TEST_CASES = [
    {
        "name": "Test 1: Legitimate",
        "email_text": (
            "Hi Sarah, can we move tomorrow's project meeting to 2:30 PM? "
            "I need a little more time to finish the quarterly report. "
            "Let me know what works for you."
        ),
        "expected_prediction": "Safe Email",
    },
    {
        "name": "Test 2: Classic Phishing",
        "email_text": (
            "URGENT: Your bank account has been suspended due to suspicious activity. "
            "Click the secure verification link below immediately to confirm your details "
            "or your account will be permanently locked."
        ),
        "expected_prediction": "Phishing Email",
    },
    {
        "name": "Test 3: Nigerian Fraud",
        "email_text": (
            "Dear Friend, I am a foreign prince seeking your assistance to transfer "
            "15 million dollars out of my country. Kindly send your bank account "
            "details so I can reward you with 30 percent of the total sum."
        ),
        "expected_prediction": "Phishing Email",
    },
    {
        "name": "Test 4: Promotional Spam",
        "email_text": (
            "Special offer just for you. Get 80% discount on premium software today only. "
            "Click here now to claim your deal before stock runs out."
        ),
        "expected_prediction": "Phishing Email",
    },
]


def send_prediction_request(email_text: str) -> dict[str, Any]:
    """Send a prediction request to the FastAPI server and return the response."""
    payload = json.dumps({"email_text": email_text}).encode("utf-8")
    request = Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    """Run the API tests and print PASS/FAIL results."""
    for test_case in TEST_CASES:
        print(test_case["name"])

        try:
            result = send_prediction_request(test_case["email_text"])
        except HTTPError as exc:
            print(f"Request failed with HTTP status {exc.code}.")
            print("FAIL\n")
            continue
        except URLError as exc:
            print(f"Could not reach the API server: {exc.reason}")
            print("FAIL\n")
            continue

        prediction = result.get("prediction", "Unknown")
        confidence = result.get("confidence", "N/A")
        phishing_probability = result.get("phishing_probability", "N/A")

        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence}")
        print(f"Phishing Probability: {phishing_probability}")

        passed = prediction == test_case["expected_prediction"]
        print("PASS\n" if passed else "FAIL\n")


if __name__ == "__main__":
    main()
