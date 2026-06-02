# Phase 1: Dataset Guide - NHANES PHQ-9 Depression Data

## Overview
National Health and Nutrition Examination Survey (NHANES) contains the PHQ-9 (Patient Health Questionnaire-9) depression screening instrument administered to participants aged 18+.

## Dataset Structure

### PHQ-9 Questions (9 items)
Each question scored 0-3 (0=Not at all, 1=Several days, 2=More than half the days, 3=Nearly every day)

| Code | Column Name | Question |
|------|-------------|----------|
| DPQ010 | Little interest or pleasure | Over last 2 weeks, have little interest in doing things? |
| DPQ020 | Feeling down/depressed | Feel down, depressed, or hopeless? |
| DPQ030 | Trouble falling asleep | Trouble falling/staying asleep or sleeping too much? |
| DPQ040 | Feeling tired | Feel tired or have little energy? |
| DPQ050 | Poor appetite | Have poor appetite or overeating? |
| DPQ060 | Self-criticism | Feel bad about yourself? |
| DPQ070 | Concentration difficulty | Have trouble concentrating? |
| DPQ080 | Slow/fast speech | Speak so slowly/quickly others notice? |
| DPQ090 | Self-harm thoughts | Thoughts that you'd be better off dead? |

### PHQ-9 Total Score
- **Calculation**: Sum of DPQ010 to DPQ090 (range: 0-27)
- **Missing Data**: If <7 items answered, PHQ9_TOTAL = null

### PHQ-9 Severity Levels
| Score | Severity | Classification |
|-------|----------|-----------------|
| 0-4   | Minimal  | No depression |
| 5-9   | Mild     | Mild depression |
| 10-14 | Moderate | Moderate depression |
| 15-19 | Moderately Severe | Moderately severe depression |
| 20-27 | Severe   | Severe depression |

## Demographic Variables

### Key Demographics (NHANES)
| Column | Description | Type | Values |
|--------|-------------|------|--------|
| SEQN | Unique identifier | Integer | 0-999999 |
| RIAGENDR | Gender | Categorical | 1=Male, 2=Female |
| RIDAGEYR | Age (years) | Integer | 18-80 |
| RIDRETH1 | Race/Ethnicity | Categorical | 1=Mexican American, 2=Other Hispanic, 3=Non-Hispanic White, 4=Non-Hispanic Black, 5=Other Race |
| INDHHIN2 | Income | Categorical | 1=<$20k, 2=$20-$44.9k, 3=$45-$74.9k, 4=$75k+ |
| DMDEDUC2 | Education | Categorical | 1=<9th grade, 2=9-11th grade, 3=High School, 4=Some College, 5=College+ |
| DMDMARTL | Marital Status | Categorical | 1=Married, 2=Widowed, 3=Divorced, 4=Separated, 5=Never married, 6=Living with partner |

### Age Groups (Created for Analysis)
- **18-25**: Young Adults
- **26-40**: Adults
- **41-55**: Middle-aged
- **56-70**: Mature Adults
- **71+**: Older Adults

## Data Access

### Option 1: Direct Download
1. Visit: https://www.cdc.gov/nchs/nhanes/
2. Select year (e.g., 2017-2018)
3. Download files:
   - Demo.txt (Demographics)
   - Dpq_xxx.txt (Depression questionnaire)
4. Save to `data/raw/`

### Option 2: NHANES API
```python
import requests
# Use NHANES API or CDC Wonder system
```

### Option 3: Synthetic Data
Use provided `data/sample_synthetic_nhanes.csv` for testing.

## Column Selection Rationale

### Included Features
✅ **PHQ-9 Components** (DPQ010-DPQ090): Direct depression screening
✅ **Gender** (RIAGENDR): Known depression disparities by gender
✅ **Age** (RIDAGEYR): Depression prevalence varies by age
✅ **Race/Ethnicity** (RIDRETH1): Important for fairness analysis
✅ **Income** (INDHHIN2): SES impacts mental health access/outcomes
✅ **Education** (DMDEDUC2): Associated with health literacy & outcomes
✅ **Marital Status** (DMDMARTL): Social support factor

### Excluded Features
❌ Medical history: Not available in all cycles
❌ Medication use: Data quality issues
❌ BMI/Physical measurements: Different questionnaire cycle

## File Structure

```
data/
├── raw/
│   ├── demographics.csv      # SEQN, RIAGENDR, RIDAGEYR, etc.
│   ├── depression_survey.csv # DPQ010-DPQ090
│   └── other_health.csv      # DMDEDUC2, INDHHIN2, DMDMARTL
├── processed/
│   ├── train.csv             # Training set (70%)
│   ├── test.csv              # Test set (30%)
│   └── validation.csv        # Optional validation set
└── sample_synthetic_nhanes.csv  # For testing/demo
```

## Data Quality Notes

- **Missing Values**: NHANES uses specific codes (7=Refused, 9=Don't know)
- **Merging**: Use SEQN (sequence number) as primary key
- **Time Period**: Ensure same survey cycle for all variables
- **Sample Design**: NHANES uses complex survey design (weights not used for model training)

## Next Steps

→ Phase 2: Preprocessing - Create `preprocess.py` to load and transform data
