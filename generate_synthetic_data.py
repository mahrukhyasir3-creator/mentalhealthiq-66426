"""
Phase 1: Synthetic NHANES PHQ-9 Dataset Generator

Generates realistic synthetic data matching NHANES structure for testing.
Production code will use real NHANES data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


def generate_synthetic_nhanes(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic NHANES-like depression screening data.

    Args:
        n_samples: Number of synthetic samples (default: 5000)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with SEQN, demographics, and PHQ-9 questions
    """
    np.random.seed(seed)

    # 1. Generate SEQN (unique identifiers)
    seqn = np.arange(1, n_samples + 1)

    # 2. Demographics
    # Gender: 1=Male, 2=Female (48/52 distribution)
    gender = np.random.choice([1, 2], size=n_samples, p=[0.48, 0.52])

    # Age: 18-80 years (realistic distribution)
    age = np.random.gamma(shape=35, scale=1.2, size=n_samples)
    age = np.clip(age, 18, 80).astype(int)

    # Race/Ethnicity (NHANES distribution)
    # 1=Mexican American, 2=Other Hispanic, 3=Non-Hispanic White,
    # 4=Non-Hispanic Black, 5=Other Race
    race = np.random.choice(
        [1, 2, 3, 4, 5], size=n_samples, p=[0.08, 0.09, 0.60, 0.13, 0.10]
    )

    # Income: 1=<$20k, 2=$20-$44.9k, 3=$45-$74.9k, 4=$75k+
    income = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.15, 0.25, 0.30, 0.30])

    # Education: 1=<9th, 2=9-11th, 3=High School, 4=Some College, 5=College+
    education = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.05, 0.06, 0.25, 0.35, 0.29])

    # Marital Status: 1=Married, 2=Widowed, 3=Divorced, 4=Separated, 5=Never married, 6=Living with partner
    marital = np.random.choice([1, 2, 3, 4, 5, 6], size=n_samples, p=[0.50, 0.05, 0.10, 0.03, 0.22, 0.10])

    # 3. PHQ-9 Questions (0=Not at all, 1=Several days, 2=More than half, 3=Nearly every day)
    # Create realistic correlation structure
    # Baseline depression tendency (0-1 scale)
    depression_tendency = np.random.beta(a=2, b=5, size=n_samples)

    # Add effects
    # Higher age -> slight increase
    age_effect = (age - 18) / 62 * 0.15
    # Female -> slight increase
    gender_effect = (gender - 1) * 0.10
    # Lower income -> increase
    income_effect = (5 - income) / 4 * 0.15
    # Lower education -> increase
    education_effect = (6 - education) / 5 * 0.10
    # Single/divorced -> increase
    single_effect = (marital > 1).astype(float) * 0.10

    # Combined depression score (0-1)
    combined_depression = np.clip(
        depression_tendency + age_effect + gender_effect + income_effect + education_effect + single_effect,
        0, 1
    )

    # Generate PHQ-9 items with correlation
    phq_items = []
    for item_num in range(9):
        # Item-specific variation
        item_bias = np.random.normal(0.3, 0.1)
        item_scores = np.random.binomial(
            n=3, p=np.clip(combined_depression + item_bias * 0.3, 0, 1), size=n_samples
        )
        phq_items.append(item_scores)

    # Ensure realistic missing values (about 2% completely missing)
    missing_mask = np.random.random(n_samples) < 0.02

    # Create DataFrame
    df = pd.DataFrame({
        'SEQN': seqn,
        'RIAGENDR': gender,
        'RIDAGEYR': age,
        'RIDRETH1': race,
        'INDHHIN2': income,
        'DMDEDUC2': education,
        'DMDMARTL': marital,
        'DPQ010': phq_items[0],
        'DPQ020': phq_items[1],
        'DPQ030': phq_items[2],
        'DPQ040': phq_items[3],
        'DPQ050': phq_items[4],
        'DPQ060': phq_items[5],
        'DPQ070': phq_items[6],
        'DPQ080': phq_items[7],
        'DPQ090': phq_items[8],
    })

    # Apply missing values
    for item in ['DPQ010', 'DPQ020', 'DPQ030', 'DPQ040', 'DPQ050', 'DPQ060', 'DPQ070', 'DPQ080', 'DPQ090']:
        df.loc[missing_mask, item] = np.nan

    return df


def main():
    """Generate and save synthetic NHANES data."""
    print("=" * 70)
    print("PHASE 1: NHANES PHQ-9 Dataset Generation")
    print("=" * 70)

    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Generate synthetic data
    print("\n1. Generating synthetic NHANES data...")
    df = generate_synthetic_nhanes(n_samples=5000, seed=42)

    # Save to CSV
    output_path = data_dir / "sample_synthetic_nhanes.csv"
    df.to_csv(output_path, index=False)
    print(f"   ✓ Saved {len(df)} samples to: {output_path}")

    # Print summary statistics
    print("\n2. Dataset Summary:")
    print(f"   - Total samples: {len(df)}")
    print(f"   - Features: {len(df.columns)}")
    print(f"   - Missing values: {df.isna().sum().sum()} ({df.isna().sum().sum() / (len(df) * len(df.columns)) * 100:.2f}%)")

    # PHQ-9 Statistics
    phq_cols = ['DPQ010', 'DPQ020', 'DPQ030', 'DPQ040', 'DPQ050', 'DPQ060', 'DPQ070', 'DPQ080', 'DPQ090']
    phq9_total = df[phq_cols].sum(axis=1)

    print("\n3. PHQ-9 Total Score Distribution:")
    print(f"   - Mean: {phq9_total.mean():.2f}")
    print(f"   - Std Dev: {phq9_total.std():.2f}")
    print(f"   - Min: {phq9_total.min():.0f}, Max: {phq9_total.max():.0f}")

    severity_counts = pd.cut(phq9_total, bins=[-1, 4, 9, 14, 19, 27], 
                            labels=['Minimal', 'Mild', 'Moderate', 'Moderately Severe', 'Severe']).value_counts()
    print("\n4. Severity Distribution:")
    for severity, count in severity_counts.items():
        pct = count / len(phq9_total) * 100
        print(f"   - {severity}: {count} ({pct:.1f}%)")

    # Demographics
    print("\n5. Demographics:")
    print(f"   - Age: Mean={df['RIDAGEYR'].mean():.1f}, Range={df['RIDAGEYR'].min()}-{df['RIDAGEYR'].max()}")
    print(f"   - Gender: Male={sum(df['RIAGENDR']==1)} ({sum(df['RIAGENDR']==1)/len(df)*100:.1f}%), "
          f"Female={sum(df['RIAGENDR']==2)} ({sum(df['RIAGENDR']==2)/len(df)*100:.1f}%)")
    print(f"   - Race: White={sum(df['RIDRETH1']==3)}, Black={sum(df['RIDRETH1']==4)}, "
          f"Other={sum(df['RIDRETH1'].isin([1,2,5]))}")

    # Sample records
    print("\n6. Sample Records:")
    print(df.head(3).to_string())

    print("\n" + "=" * 70)
    print("NEXT STEP: Phase 2 - Preprocessing (preprocess.py)")
    print("=" * 70)


if __name__ == "__main__":
    main()
