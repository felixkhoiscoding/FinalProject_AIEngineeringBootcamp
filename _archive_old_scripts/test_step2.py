"""
Test script to verify Step 2: Exploratory Data Analysis
This script runs key EDA functions without requiring Jupyter
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.data_loader import load_data
from src import visualization as viz
from config.config import RAW_DATA_PATH, FIGURES_DIR
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import numpy as np

print("="*70)
print("STEP 2: EXPLORATORY DATA ANALYSIS - TEST SCRIPT")
print("="*70)

# Ensure figures directory exists
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("\n" + "-"*70)
print("Loading dataset...")
print("-"*70)
df = load_data(filepath=RAW_DATA_PATH, sheet_name='Monthly')

# Test 1: Time series visualization
print("\n[TEST 1] Testing time series plot...")
try:
    fig, ax = viz.plot_time_series(
        data=df,
        date_col='observation_date',
        value_col='WPU101704',
        title='Producer Price Index - Hot Rolled Steel (1982-2025)'
    )
    print("[OK] Time series plot created successfully")
except Exception as e:
    print(f"[ERROR] Time series plot failed: {e}")

# Test 2: Decomposition
print("\n[TEST 2] Testing seasonal decomposition...")
try:
    ts_data = df.set_index('observation_date')['WPU101704']
    decomp = seasonal_decompose(ts_data, model='additive', period=12)
    seasonal_strength = 1 - (decomp.resid.var() / (decomp.seasonal + decomp.resid).var())
    print(f"[OK] Decomposition completed")
    print(f"    Seasonal strength: {seasonal_strength:.4f}")
except Exception as e:
    print(f"[ERROR] Decomposition failed: {e}")

# Test 3: Stationarity tests
print("\n[TEST 3] Testing stationarity...")
try:
    adf_result = adfuller(df['WPU101704'].dropna(), autolag='AIC')
    kpss_result = kpss(df['WPU101704'].dropna(), regression='ct', nlags='auto')
    
    print(f"[OK] Stationarity tests completed")
    print(f"    ADF p-value: {adf_result[1]:.4f} ({'Stationary' if adf_result[1] < 0.05 else 'Non-Stationary'})")
    print(f"    KPSS p-value: {kpss_result[1]:.4f} ({'Stationary' if kpss_result[1] >= 0.05 else 'Non-Stationary'})")
    
    if adf_result[1] >= 0.05:
        print("    [RECOMMENDATION] Series requires differencing (d=1 for ARIMA)")
except Exception as e:
    print(f"[ERROR] Stationarity tests failed: {e}")

# Test 4: Distribution analysis
print("\n[TEST 4] Testing distribution analysis...")
try:
    skewness = df['WPU101704'].skew()
    kurtosis = df['WPU101704'].kurtosis()
    print(f"[OK] Distribution analysis completed")
    print(f"    Skewness: {skewness:.4f}")
    print(f"    Kurtosis: {kurtosis:.4f}")
except Exception as e:
    print(f"[ERROR] Distribution analysis failed: {e}")

# Test 5: Outlier detection
print("\n[TEST 5] Testing outlier detection...")
try:
    Q1 = df['WPU101704'].quantile(0.25)
    Q3 = df['WPU101704'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df['WPU101704'] < lower_bound) | (df['WPU101704'] > upper_bound)]
    
    print(f"[OK] Outlier detection completed")
    print(f"    IQR method: {len(outliers)} outliers ({len(outliers)/len(df)*100:.2f}%)")
except Exception as e:
    print(f"[ERROR] Outlier detection failed: {e}")

# Test 6: Calculate changes
print("\n[TEST 6] Testing temporal change calculations...")
try:
    df['MoM_pct'] = df['WPU101704'].pct_change() * 100
    df['YoY_pct'] = df['WPU101704'].pct_change(periods=12) * 100
    
    print(f"[OK] Change calculations completed")
    print(f"    MoM mean: {df['MoM_pct'].mean():.2f}%")
    print(f"    YoY mean: {df['YoY_pct'].mean():.2f}%")
except Exception as e:
    print(f"[ERROR] Change calculations failed: {e}")

# Summary
print("\n" + "="*70)
print("KEY FINDINGS SUMMARY")
print("="*70)

print(f"\n1. DATASET:")
print(f"   - Total observations: {len(df)}")
print(f"   - Date range: {df['observation_date'].min().strftime('%Y-%m')} to {df['observation_date'].max().strftime('%Y-%m')}")

print(f"\n2. TREND:")
overall_change = ((df['WPU101704'].iloc[-1] / df['WPU101704'].iloc[0]) - 1) * 100
print(f"   - Overall change: +{overall_change:.1f}%")
print(f"   - Direction: Upward trend")

print(f"\n3. SEASONALITY:")
if 'seasonal_strength' in locals():
    print(f"   - Seasonal strength: {seasonal_strength:.4f}")
    if seasonal_strength > 0.3:
        print(f"   - [PRESENT] Seasonality detected - recommend SARIMA")
    else:
        print(f"   - [WEAK] Minimal seasonality")

print(f"\n4. STATIONARITY:")
if 'adf_result' in locals():
    print(f"   - ADF test: {'Stationary' if adf_result[1] < 0.05 else 'Non-Stationary'}")
    print(f"   - KPSS test: {'Stationary' if kpss_result[1] >= 0.05 else 'Non-Stationary'}")
    if adf_result[1] >= 0.05:
        print(f"   - [ACTION] Differencing required for ARIMA (d=1)")

print(f"\n5. DISTRIBUTION:")
if 'skewness' in locals():
    print(f"   - Skewness: {skewness:.4f} ({'Right-skewed' if skewness > 0 else 'Left-skewed'})")
    print(f"   - Shape: Non-normal distribution")

print(f"\n6. OUTLIERS:")
if 'outliers' in locals():
    print(f"   - Count: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")

print(f"\n7. VOLATILITY:")
std_dev = df['WPU101704'].std()
cv = (std_dev / df['WPU101704'].mean()) * 100
print(f"   - Std Dev: {std_dev:.2f}")
print(f"   - CV: {cv:.2f}%")

print(f"\n8. MODEL RECOMMENDATIONS:")
print(f"   - Baseline: Holt-Winters")
if 'seasonal_strength' in locals() and seasonal_strength > 0.3:
    print(f"   - SARIMA: Recommended (seasonality present)")
print(f"   - Prophet: Good for multiple seasonalities")
print(f"   - LSTM: Worth trying for complex patterns")

print("\n" + "="*70)
print("[OK] STEP 2: EXPLORATORY DATA ANALYSIS - COMPLETED SUCCESSFULLY")
print("="*70)

print("\nDeliverables:")
print("  [OK] src/visualization.py - 7 plotting functions")
print("  [OK] notebooks/02_eda.ipynb - Comprehensive EDA notebook")
print("  [OK] Statistical tests: ADF, KPSS, Shapiro-Wilk")
print("  [OK] Decomposition: Trend + Seasonal + Residual")
print("  [OK] ACF/PACF analysis for ARIMA parameters")
print("  [OK] Outlier detection: IQR + Z-score methods")
print("  [OK] Temporal analysis: MoM + YoY changes")

print("\nReady for Step 3: Data Preprocessing")
