import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import seaborn as sns

import random
# 设置随机种子
SEED = 42 # 复现结果
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

def load_and_preprocess_data(afs_file, theta_file):
    afs_data = []
    max_length = 0
    
    with open(afs_file, 'r') as file:
        for line in file:
            values = line.strip().split()
            length = len(values)
            max_length = max(max_length, length)
    
    with open(afs_file, 'r') as file:
        for line in file:
            values = list(map(float, line.strip().split()))
            if len(values) < max_length:
                values.extend([0.0] * (max_length - len(values)))
            afs_data.append(values)
    
    with open(theta_file, 'r') as file:
        theta_data = [float(line.strip()) for line in file]
    
    assert len(afs_data) == len(theta_data), f"AFS and Theta data lengths don't match! AFS: {len(afs_data)}, Theta: {len(theta_data)}"
    
    X = np.array(afs_data)
    y = np.array(theta_data).reshape(-1, 1)
    
    print(f"Data shapes: AFS={X.shape}, Theta={y.shape}")
    
    return X, y

def plot_training_history(history, title="Model Training History"):
    plt.figure(figsize=(12, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    plt.show()

def normalized_mse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    
    squared_error = np.square(y_true[mask] - y_pred[mask])
    squared_true = np.square(y_true[mask])
    
    return np.mean(squared_error / squared_true)

def normalized_mae(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mask = y_true != 0
    
    if not np.any(mask):
        return np.nan
    
    absolute_error = np.abs(y_true[mask] - y_pred[mask])
    absolute_true = np.abs(y_true[mask])
    
    return np.mean(absolute_error / absolute_true)

def plot_predictions(y_true, y_pred, dataset_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    correlation_percentage = np.corrcoef(y_true.flatten(), y_pred.flatten())[0,1]
    
    nmae = normalized_mae(y_true.flatten(), y_pred.flatten())
    nmse = normalized_mse(y_true.flatten(), y_pred.flatten())
    
    plt.figure(figsize=(10, 6))
    
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='w', linewidth=0.5)
    
    min_val = min(y_true.min(), y_pred.min()) * 1
    max_val = max(y_true.max(), y_pred.max()) * 1.05

    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    # 对角线和拟合曲线的制作
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y = x')
    z = np.polyfit(y_true.flatten(), y_pred.flatten(), 1)
    p = np.poly1d(z)
    plt.plot(y_true.flatten(), p(y_true.flatten()), "b-", alpha=0.7)
    
    result_text = (f"Normalized MSE: {nmse:.4f}\nNormalized MAE: {nmae:.4f}\nR² score: {r2:.4f}\nCorrelation: {correlation_percentage:.4f}")
    plt.text(0.05, 0.95, # 调整方块的位置
             result_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='none', pad=0.5),
             color='white')
    
    # set legend
    plt.title(f'Actual vs Predicted ({dataset_name})')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{dataset_name}_predictions.png')
    plt.show()
    
    print(f"\nResult for {dataset_name}:")
    print(f"Normalized MSE: {nmse:.4f}, Normalized MAE: {nmae:.4f}, R² score: {r2:.4f}, Correlation: {correlation_percentage:.4f}")
    
    return mse, mae, r2, correlation_percentage


def build_and_train_model(X_train, y_train, X_val, y_val):
    X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_val_reshaped = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
    
    model = Sequential([
        LSTM(16, activation='tanh', return_sequences=True, input_shape=(1, X_train.shape[1]),
            kernel_regularizer=l2(0.02),
            recurrent_regularizer=l2(0.01),
            activity_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.6),
        LSTM(8, activation='tanh',
            kernel_regularizer=l2(0.02), 
            recurrent_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.6),
        Dense(4, activation='relu', 
            kernel_regularizer=l2(0.02)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1)
    ])
        
    model.compile(optimizer= keras.optimizers.Adam(learning_rate=0.0005), loss='mse')
    
    model.summary()
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=1)
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)
    
    history = model.fit(
        X_train_reshaped, y_train,
        validation_data=(X_val_reshaped, y_val),
        epochs=150,
        batch_size=32,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return model, history


afs_train_file = 'Mar1_data\\AFS_train_filled.txt'
theta_train_file = 'Mar1_data\\theta_train.txt'

afs_val_file = 'Mar1_data\\AFS_valid_filled.txt'
theta_val_file = 'Mar1_data\\theta_valid.txt'

afs_test_file = 'Mar1_data\\AFS_test_filled.txt'
theta_test_file = 'Mar1_data\\theta_test.txt'

X_train_raw, y_train_raw = load_and_preprocess_data(afs_train_file, theta_train_file)
X_val_raw, y_val_raw = load_and_preprocess_data(afs_val_file, theta_val_file)
X_test_raw, y_test_raw = load_and_preprocess_data(afs_test_file, theta_test_file)

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train_raw)
X_val_scaled = scaler_X.transform(X_val_raw)
X_test_scaled = scaler_X.transform(X_test_raw)

scaler_y = StandardScaler()
y_train_scaled = scaler_y.fit_transform(y_train_raw)
y_val_scaled = scaler_y.transform(y_val_raw)
y_test_scaled = scaler_y.transform(y_test_raw)

model, history = build_and_train_model(
    X_train_scaled, y_train_scaled,
    X_val_scaled, y_val_scaled
)

print(plot_training_history(history, "LSTM Model Training History"))

X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_val_reshaped = X_val_scaled.reshape(X_val_scaled.shape[0], 1, X_val_scaled.shape[1])
X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

train_pred = model.predict(X_train_reshaped)
val_pred = model.predict(X_val_reshaped)
test_pred = model.predict(X_test_reshaped)

train_pred_orig = scaler_y.inverse_transform(train_pred)
val_pred_orig = scaler_y.inverse_transform(val_pred)
test_pred_orig = scaler_y.inverse_transform(test_pred)

train_results = plot_predictions(y_train_raw, train_pred_orig, "LSTM_Training_Set")
val_results = plot_predictions(y_val_raw, val_pred_orig, "LSTM_Validation_Set")
test_results = plot_predictions(y_test_raw, test_pred_orig, "LSTM_Test_Set")
