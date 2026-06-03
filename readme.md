# MentalHealthIQ

Depression severity prediction API using NHANES dataset.

## Live URL
> Coming soon — will be updated after deployment

## CI Status
> Badge coming after CI/CD setup

## Tech Stack
- Logistic Regression / Random Forest / optional XGBoost + optional SMOTE
- FastAPI
- MongoDB Atlas / Local MongoDB
- Docker
- Hugging Face Spaces
- GitHub Actions

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
   - `http://localhost:8000/health`

## What The App Does
MentalHealthIQ is a clinic-style depression screening assistant. Clinic staff fills a PHQ-9 form, the FastAPI backend predicts depression severity, and the doctor sees:

- risk percentage
- severity label
- recommendation
- warning message
- fairness note
- printable patient report
- saved history when MongoDB is configured
- comparison between current and previous visits

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
MongoDB is optional for local demos. If MongoDB is not configured, prediction still works, but history, comparison, and saved reports return empty states.

To enable saved history, set:

```powershell
MONGODB_URI=your_mongodb_connection_string
MONGODB_DATABASE=mentalhealthiq
MONGODB_COLLECTION=predictions
```

## Local MongoDB With Compass
The backend defaults to local MongoDB:

```env
MONGODB_URI=mongodb://localhost:27017
MONGODB_DATABASE=mentalhealthiq
MONGODB_COLLECTION=predictions
```

1. Start MongoDB.

Option A, installed MongoDB service:

```powershell
net start MongoDB
```

Option B, Docker Compose:

```powershell
docker compose up -d mongodb mongo-express
```

2. Start the API:

```powershell
python -m uvicorn mentalhealthiq.api:app --reload --port 8000
```

3. Check health:

```text
http://localhost:8000/health
```

Expected:

```json
{"mongo": "ready"}
```

4. Open MongoDB Compass:

```text
mongodb://localhost:27017
```

5. Submit a prediction with `Predict and Save`.

6. Check:

```text
mentalhealthiq -> predictions
```

If FastAPI runs inside Docker Compose, use:

```env
MONGODB_URI=mongodb://mongodb:27017
```

## Local MongoDB Compass Setup
1. Start your local MongoDB server.
2. Open MongoDB Compass and connect to:
   - `mongodb://localhost:27017`
3. Use database: `mentalhealthiq`
4. Use collection: `predictions`
5. Copy this file to `.env` if needed and keep the repository root as the working directory.
