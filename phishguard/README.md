# PhishGuard — Phishing Email Detection System

PhishGuard is a production-ready phishing email detection system that classifies email text as safe or phishing using TF-IDF features and an XGBoost classifier. The FastAPI service exposes single-email prediction, batch prediction, health checks, model metadata, and explanatory risk flags for common phishing signals.

## Model Performance

| Model | Dataset | Accuracy | Precision | Recall | F1 Score |
| --- | --- | ---: | ---: | ---: | ---: |
| v1 | Original cleaned dataset | Not recorded | Not recorded | Not recorded | Not recorded |
| v2.0 | Combined dataset, 179,052 emails | 98.01% | 97.00% | 99.18% | 98.08% |

## Folder Structure

```text
phishguard/
├── api/
│   ├── main.py
│   └── models/
│       ├── xgboost_model.pkl
│       └── tfidf_vectorizer.pkl
├── ml/
│   ├── merge_datasets.py
│   ├── retrain_model.py
│   └── generate_visualizations.py
├── data/
│   ├── combined_emails.csv
│   └── cleaned_emails.csv
├── visualizations/
├── tests/
│   └── test_api.py
├── requirements.txt
└── README.md
```

## Run The API

```bash
cd phishguard
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python api/main.py
```

The API runs on `http://localhost:8000`.

## Deploy To Render

1. Push this project to GitHub.
2. In Render, create a new **Web Service** from the GitHub repository.
3. If your repository contains this project inside a `phishguard/` folder, set **Root Directory** to `phishguard`.
4. Use these Render settings:

```text
Environment: Python
Build Command: pip install -r requirements.txt
Start Command: python api/main.py
Health Check Path: /health
```

The API reads Render's `PORT` environment variable automatically. After deployment, test:

```bash
curl https://your-render-service.onrender.com/health
curl https://your-render-service.onrender.com/model-info
```

The deployed API only needs `api/main.py`, `api/models/`, and `requirements.txt`. Large training CSVs are ignored by `.gitignore` because they are not needed on Render and may exceed GitHub's file-size limits.

## Retrain The Model

```bash
cd phishguard
source .venv/bin/activate
python ml/retrain_model.py
```

The retraining script reads `data/combined_emails.csv` and writes the updated model artifacts to `api/models/`.

## Generate Evaluation Visualizations

```bash
cd phishguard
source .venv/bin/activate
python ml/generate_visualizations.py
```

Generated PNG files are saved to `visualizations/`.

## Tech Stack

- Python
- FastAPI
- XGBoost
- TF-IDF
- Flutter
