# MentalHealthIQ

Clinic-style PHQ-9 depression severity screening using NHANES-style demographic and questionnaire data.

## Important Caveat

MentalHealthIQ is a PHQ-9 screening and severity classification demo. The current model uses the nine PHQ-9 answers as input features while the severity label is derived from the PHQ-9 total score. This is useful for demonstrating the end-to-end workflow, but it is not an independent clinical diagnosis model.

API prediction responses use standard PHQ-9 scoring boundaries as the primary `predicted_severity`. The trained ML output is returned separately as `model_predicted_severity` for supporting analytics/demo context.

## Quick Start

### Windows PowerShell

```powershell
.\scripts\setup.ps1
# place demographic.csv and questionnaire.csv in data/raw
.\scripts\bootstrap.ps1
.\scripts\run-all.ps1
```

If PowerShell blocks local scripts, run this once in the same terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then open:

```text
API:           http://localhost:8000
Health:        http://localhost:8000/health
Frontend:      http://localhost:5500
Mongo Express: http://localhost:8081
```

### Manual Fallback

```powershell
pip install -r requirements.txt
python scripts/bootstrap_ml.py
python -m uvicorn mentalhealthiq.api:app --reload --port 8000
python -m http.server 5500 -d frontend
```

### Docker

If Docker is installed, the full service stack can be started with:

```powershell
docker compose up --build
```

The API container uses `mongodb://mongodb:27017` internally. For predictions to work, model artifacts must exist in `data/`; run `.\scripts\bootstrap.ps1` first when starting from raw CSVs only.

The repository may include a small generated `data/models/model.joblib` so the API can start with an existing demo model. Re-run bootstrap whenever raw data changes or artifacts are missing.

### Linux/macOS Helpers

```bash
chmod +x scripts/*.sh
./scripts/setup.sh
# place demographic.csv and questionnaire.csv in data/raw
./scripts/bootstrap.sh
./scripts/run-all.sh
```

## Tech Stack

- FastAPI
- scikit-learn Logistic Regression / Random Forest
- Optional XGBoost
- Optional SMOTE with imbalanced-learn
- MongoDB Atlas or local MongoDB
- Plain HTML/CSS/JavaScript frontend
- Chart.js
- Docker Compose for MongoDB, Mongo Express, API, and frontend

## Dataset Setup

1. Extract the downloaded dataset zip.
2. Copy only:
   - `demographic.csv`
   - `questionnaire.csv`
3. Paste them into:
   - `data/raw/demographic.csv`
   - `data/raw/questionnaire.csv`
4. Run:

   ```powershell
   python scripts/bootstrap_ml.py
   ```

5. Start API:

   ```powershell
   python -m uvicorn mentalhealthiq.api:app --reload --port 8000
   ```

6. Open:

   ```text
   http://localhost:8000/health
   ```

## What The App Does

MentalHealthIQ lets clinic staff fill a PHQ-9 form, submit it to the FastAPI backend, and view:

- PHQ-9 total score
- severity label
- risk percentage and risk band
- recommendation
- warning message
- fairness note
- printable patient report
- saved history when MongoDB is configured
- comparison between current and previous visits
- model metrics and fairness reports

## ML Pipeline

Run the full pipeline with:

```powershell
python scripts/bootstrap_ml.py
```

This generates:

- `data/processed/train.csv`
- `data/processed/test.csv`
- `data/processed/train_raw.csv`
- `data/processed/test_raw.csv`
- `data/processed/preprocessor.joblib`
- `data/models/model.joblib`
- `data/fairness_reports/fairness_report.csv`

## API Endpoints

- `GET /health` - system status
- `POST /predict` - predict only, without saving
- `POST /predict-and-save` - predict and save to MongoDB
- `GET /predictions` - saved prediction history
- `GET /patients/{patient_id}/history` - one patient's saved visits
- `GET /patients/{patient_id}/comparison` - compare latest and previous visit
- `GET /stats` - dashboard summary numbers
- `GET /fairness-report` - raw fairness report
- `GET /fairness-summary` - simple fairness summary for the UI
- `GET /metrics` - model performance

## Frontend Usage

Open `frontend/index.html` in a browser, or serve the folder with any simple static server.

Main pages:

- Home
- Dashboard
- PHQ-9 Patient Form
- Prediction Result
- Patient History
- Patient Comparison
- Fairness Analysis
- Reports
- Metrics
- System Health

Reports use browser printing. Open the Reports page and click `Print Report`.

## MongoDB

MongoDB is optional for local demos. If MongoDB is unavailable, prediction still works, but history, comparison, dashboard history, and saved reports return empty/default states.

Defaults:

```env
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=mentalhealthiq
MONGODB_COLLECTION=predictions
```

To enable saved history, set the same values in `.env` or your environment.

## Local MongoDB With Docker

Start MongoDB and Mongo Express:

```powershell
docker compose up -d mongodb mongo-express
```

Open Mongo Express:

```text
http://localhost:8081
```

Open MongoDB Compass:

```text
mongodb://localhost:27017
```

Use:

```text
mentalhealthiq -> predictions
```

If FastAPI runs inside Docker Compose, use:

```env
MONGODB_URI=mongodb://mongodb:27017
```

## Tests

Run:

```powershell
python -m pytest
```
