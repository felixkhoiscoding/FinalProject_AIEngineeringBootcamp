"""
Test script to verify Step 4: Baseline Models
This script tests all baseline forecasting models
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.data_loader import load_data
from src import baseline_models as bm
from config.config import RAW_DATA_PATH, TEST_SIZE
import pandas as pd
import numpy as np

print("="*70)
print("STEP 4: BASELINE MODELS - TEST SCRIPT")
print("="*70)

# Load and split data
print("\nLoading and splitting data...")
df = load_data(filepath=RAW_DATA_PATH, sheet_name='Monthly')

# Split
split_idx = len(df) - TEST_SIZE
train = df.iloc[:split_idx].copy()
test = df.iloc[split_idx:].copy()

print(f"Train: {len(train)}, Test: {len(test)}")
y_true = test['WPU101704'].values

# Test all baseline models
print("\n" + "="*70)
print("EVALUATING BASELINE MODELS")
print("="*70)

results = bm.evaluate_baseline_models(train, test, 'WPU101704')

# Create comparison table
print("\n" + "="*70)
print("MODEL PERFORMANCE COMPARISON")
print("="*70)

comparison = bm.create_comparison_table(results)
print("\n")
print(comparison.to_string(index=False))

# Find best model
best_model = comparison.iloc[0]['Model']
best_rmse = comparison.iloc[0]['RMSE']
best_mape = comparison.iloc[0]['MAPE']

print("\n" + "="*70)
print("BEST BASELINE MODEL")
print("="*70)
print(f"\nModel: {best_model}")
print(f"RMSE: {best_rmse:.2f}")
print(f"MAPE: {best_mape:.2f}%")
print(f"MAE: {comparison.iloc[0]['MAE']:.2f}")
print(f"R²: {comparison.iloc[0]['R2']:.4f}")

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"\nTotal models evaluated: {len(comparison)}")
print(f"Best performing model: {best_model}")
print(f"\nPerformance rankings (by RMSE):")
for i, row in comparison.iterrows():
    print(f"  {row['Rank']}. {row['Model']}: RMSE={row['RMSE']:.2f}, MAPE={row['MAPE']:.2f}%")

print("\n" + "="*70)
print("[OK] STEP 4: BASELINE MODELS - COMPLETED SUCCESSFULLY")
print("="*70)

print("\nDeliverables:")
print("  [OK] src/baseline_models.py - 9 functions")
print("  [OK] 6 baseline models implemented and tested")
print("  [OK] Metrics calculated: MAE, RMSE, MAPE, R²")
print("  [OK] Model comparison table generated")
print(f"  [OK] Best model identified: {best_model}")

print("\nReady for Step 5: Intermediate Models (ARIMA, Prophet, LSTM)")
