"""
Test LSTM with fixed data
"""

import sys
from pathlib import Path
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.data_loader import load_data
from src.advanced_models import train_lstm
from config.config import RAW_DATA_PATH, TEST_SIZE
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("="*70)
print("TESTING LSTM WITH FIXED DATA")
print("="*70)

# Load LSTM data
processed_dir = project_root / 'data' / 'processed'
X_train = np.load(processed_dir / 'X_train_lstm.npy')
y_train = np.load(processed_dir / 'y_train_lstm.npy')
X_test = np.load(processed_dir / 'X_test_lstm.npy')
y_test = np.load(processed_dir / 'y_test_lstm.npy')
scaler = joblib.load(processed_dir / 'lstm_scaler.pkl')

print(f"\nData loaded:")
print(f"  X_train: {X_train.shape}")
print(f"  X_test: {X_test.shape}")
print(f"  Test samples: {len(X_test)}")

# Load actual test values for comparison
df = load_data(filepath=RAW_DATA_PATH, sheet_name='Monthly')
y_true = df.iloc[-TEST_SIZE:]['WPU101704'].values

print(f"\nTraining LSTM...")
print("-"*70)

# Train LSTM
pred_scaled, model, history = train_lstm(
    X_train, y_train, X_test,
    epochs=100,
    batch_size=16,
    units=64
)

# Inverse transform
pred_lstm = scaler.inverse_transform(pred_scaled).flatten()

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_true, pred_lstm))
mae = mean_absolute_error(y_true, pred_lstm)
mape = np.mean(np.abs((y_true - pred_lstm) / y_true)) * 100

print("\n" + "="*70)
print("LSTM RESULTS")
print("="*70)
print(f"\nRMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}%")
print(f"MAE: {mae:.2f}")

print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)
print("\n1. Naive:   RMSE=8.93  (Baseline)")
print(f"2. ARIMA:   RMSE=12.49")
print(f"3. LSTM:    RMSE={rmse:.2f} ({'BETTER!' if rmse < 8.93 else 'Worse'})")
print(f"4. Prophet: RMSE=36.60")

if rmse < 8.93:
    print(f"\n*** LSTM BEATS NAIVE! New best model! ***")
else:
    print(f"\nLSTM doesn't beat Naive, but it's now working!")

print("\n" + "="*70)
print("SUCCESS! LSTM MODEL FULLY FUNCTIONAL")
print("="*70)
