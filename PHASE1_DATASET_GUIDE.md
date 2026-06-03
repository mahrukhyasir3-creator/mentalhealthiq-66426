# Dataset Guide - NHANES PHQ-9 Depression Data

MentalHealthIQ uses real NHANES-style CSV files only.

## Required Files

Place these files in `data/raw/`:

- `data/raw/demographic.csv`
- `data/raw/questionnaire.csv`

Ignore these files for the current version:

- `diet.csv`
- `examination.csv`
- `labs.csv`
- `medications.csv`

## Required Columns

`demographic.csv` must contain:

- `SEQN`
- `RIDAGEYR`
- `RIAGENDR`
- `RIDRETH1`
- `INDHHIN2`
- `DMDEDUC2`
- `DMDMARTL`

`questionnaire.csv` must contain:

- `SEQN`
- `DPQ010`
- `DPQ020`
- `DPQ030`
- `DPQ040`
- `DPQ050`
- `DPQ060`
- `DPQ070`
- `DPQ080`
- `DPQ090`

## PHQ-9 Cleaning

Valid PHQ-9 item values are `0`, `1`, `2`, and `3`.

These values are treated as missing or invalid:

- `7`
- `9`
- `77`
- `99`
- blank values
- `NaN`
- invalid strings
- any value outside `0`, `1`, `2`, or `3`

Rows are dropped when any PHQ-9 item is missing after cleaning.

## Derived Fields

`PHQ9_TOTAL` is the sum of `DPQ010` through `DPQ090`.

`SEVERITY` is created from `PHQ9_TOTAL`:

- `0-4`: Minimal
- `5-9`: Mild
- `10-14`: Moderate
- `15-19`: Moderately Severe
- `20-27`: Severe

`AGE_GROUP` is created from `RIDAGEYR`:

- age `< 18`: Under 18
- `18-29`
- `30-44`
- `45-59`
- `60+`

## Local Setup

1. Extract the downloaded dataset zip.
2. Copy only `demographic.csv` and `questionnaire.csv`.
3. Paste them into `data/raw/`.
4. Run:

```powershell
python scripts/bootstrap_ml.py
```

5. Start the API:

```powershell
python -m uvicorn mentalhealthiq.api:app --reload --port 8000
```

6. Open:

```text
http://localhost:8000/health
```
