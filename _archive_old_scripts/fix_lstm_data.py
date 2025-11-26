"""
Create LSTM test data using walk-forward approach
Solution: Use last 12 months of training as context for test predictions
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

project_root = Path(__file__).parent
sys.path.append(str(project_root))

from src.data_loader import load_data
from config.config import RAW_DATA_PATH, TEST_SIZE, LSTM_SEQUENCE_LENGTH

print("="*70)
print("FIXING LSTM TEST DATA")
print("="*70)

# Load data
df = load_data(filepath=RAW_DATA_PATH, sheet_name='Monthly')
split_idx = len(df) - TEST_SIZE
train = df.iloc[:split_idx].copy()
test = df.iloc[split_idx:].copy()

print(f"\nOriginal split: {len(train)} train, {len(test)} test")
print(f"Problem: Test size ({len(test)}) = Sequence length ({LSTM_SEQUENCE_LENGTH})")
print(f"Result: 0 test sequences\n")

print("SOLUTION: Use last {LSTM_SEQUENCE_LENGTH} months of training as context")
print("-"*70)

# Combine last part of train with test for creating sequences
# We need sequence_length months before each test month
combined = pd.concat([
    train.tail(LSTM_SEQUENCE_LENGTH),  # Last 12 months of training
    test  # All 12 test months
])

print(f"\nCombined data: {len(combined)} months")
print(f"  - Context (from train): {LSTM_SEQUENCE_LENGTH} months")
print(f"  - Predictions (test): {len(test)} months")

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))

# Fit scaler on TRAINING data only (important!)
scaler.fit(train[['WPU101704']])

# Transform combined data
combined_scaled = scaler.transform(combined[['WPU101704']])

# Create sequences for testing
X_test, y_test = [], []

for i in range(LSTM_SEQUENCE_LENGTH, len(combined_scaled)):
    X_test.append(combined_scaled[i-LSTM_SEQUENCE_LENGTH:i, 0])
    y_test.append(combined_scaled[i, 0])

X_test = np.array(X_test)
y_test = np.array(y_test)

# Reshape for LSTM
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"\nLSTM Test Sequences Created:")
print(f"  X_test shape: {X_test.shape} (samples, timesteps, features)")
print(f"  y_test shape: {y_test.shape}")
print(f"  Number of test predictions: {len(y_test)}")

# Create training data too
train_scaled = scaler.transform(train[['WPU101704']])

X_train, y_train = [], []
for i in range(LSTM_SEQUENCE_LENGTH, len(train_scaled)):
    X_train.append(train_scaled[i-LSTM_SEQUENCE_LENGTH:i, 0])
    y_train.append(train_scaled[i, 0])

X_train = np.array(X_train)
y_train = np.array(y_train)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

print(f"\nLSTM Training Sequences:")
print(f"  X_train shape: {X_train.shape}")
print(f"  y_train shape: {y_train.shape}")

# Save the corrected data
processed_dir = project_root / 'data' / 'processed'
processed_dir.mkdir(parents=True, exist_ok=True)

np.save(processed_dir / 'X_train_lstm.npy', X_train)
np.save(processed_dir / 'y_train_lstm.npy', y_train)
np.save(processed_dir / 'X_test_lstm.npy', X_test)
np.save(processed_dir / 'y_test_lstm.npy', y_test)
joblib.dump(scaler, processed_dir / 'lstm_scaler.pkl')

print(f"\n[OK] Saved corrected LSTM data to: {processed_dir}")
print("="*70)
print("LSTM DATA FIXED!")
print("="*70)
print("\nFiles saved:")
print("  - X_train_lstm.npy")
print("  - y_train_lstm.npy")  
print("  - X_test_lstm.npy (NOW HAS DATA!)")
print("  - y_test_lstm.npy (NOW HAS DATA!)")
print("  - lstm_scaler.pkl")
print(f"\nLSTM can now be evaluated on {len(y_test)} test samples!")
