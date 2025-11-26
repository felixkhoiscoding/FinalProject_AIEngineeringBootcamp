"""
Test Step 5: Advanced Models (ARIMA, Prophet, LSTM)
Target: Beat baseline RMSE < 8.93
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.data_loader import load_data
from src import advanced_models as adv
from config.config import RAW_DATA_PATH, TEST_SIZE
import pandas as pd
import numpy as np
import joblib

print("="*70)
print("STEP 5: INTERMEDIATE MODELS - TEST SCRIPT")
print("="*70)

# Load data
df = load_data(filepath=RAW_DATA_PATH, sheet_name='Monthly')
split_idx = len(df) - TEST_SIZE
train = df.iloc[:split_idx].copy()
test = df.iloc[split_idx:].copy()

print(f"\nData: {len(train)} train, {len(test)} test")

# Load LSTM data
processed_dir = project_root / 'data' / 'processed'
try:
    X_train_lstm = np.load(processed_dir / 'X_train_lstm.npy')
    y_train_lstm = np.load(processed_dir / 'y_train_lstm.npy')
    X_test_lstm = np.load(processed_dir / 'X_test_lstm.npy')
    y_test_lstm = np.load(processed_dir / 'y_test_lstm.npy')
    scaler = joblib.load(processed_dir / 'lstm_scaler.pkl')
    print(f"LSTM data loaded: {X_train_lstm.shape}, {X_test_lstm.shape}")
except:
    print("[WARNING] LSTM data not found, will skip LSTM")
    X_train_lstm = X_test_lstm = y_train_lstm = y_test_lstm = scaler = None

# Evaluate all models
print("\n" + "="*70)
print("EVALUATING ADVANCED MODELS")
print("="*70)

results = adv.evaluate_advanced_models(
    train, test,
    X_train_lstm, y_train_lstm,
    X_test_lstm, y_test_lstm,
    scaler
)

# Compare with baseline
print("\n" + "="*70)
print("RESULTS COMPARISON")
print("="*70)
print(f"\nBASELINE (Naive): RMSE=8.93, MAPE=2.89%")
print("\nADVANCED MODELS:")

comparison = []
for name, res in results.items():
    if res is not None:
        print(f"\n{name}:")
        print(f"  RMSE: {res['RMSE']:.2f} ({'BETTER' if res['RMSE'] < 8.93 else 'WORSE'} than baseline)")
        print(f"  MAPE: {res['MAPE']:.2f}%")
        print(f"  MAE: {res['MAE']:.2f}")
        comparison.append({
            'Model': name,
            'RMSE': res['RMSE'],
            'MAPE': res['MAPE'],
            'MAE': res['MAE']
        })

# Summary
comp_df = pd.DataFrame(comparison).sort_values('RMSE')
print("\n" + "="*70)
print("FINAL RANKINGS")
print("="*70)
print("\n" + comp_df.to_string(index=False))

best = comp_df.iloc[0]
print(f"\n✓ Best Model: {best['Model']}")
print(f"  RMSE: {best['RMSE']:.2f}")
print(f"  MAPE: {best['MAPE']:.2f}%")

print("\n" + "="*70)
print("[OK] STEP 5 COMPLETED")
print("="*70)
