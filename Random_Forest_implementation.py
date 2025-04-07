import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

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

def train_baseline_models(X_train, y_train, X_val, y_val, X_test, y_test):
    results = {}
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=4,
        max_features='sqrt',
        oob_score=True,
        random_state=42 # 同样sample 复现
    )

    rf_model.fit(X_train, y_train)
    
    rf_train_pred = rf_model.predict(X_train)
    rf_val_pred = rf_model.predict(X_val)
    rf_test_pred = rf_model.predict(X_test)

    importances = rf_model.feature_importances_
    
    results['random_forest'] = {
        'model': rf_model,
        'train_pred': rf_train_pred,
        'val_pred': rf_val_pred,
        'test_pred': rf_test_pred,
        'feature_importance': importances
    }
    
    return results

def normalized_mse(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    mask = y_true != 0
    # print(mask)
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
    correlation_percentage = np.corrcoef(y_true.flatten(),y_pred.flatten()) [0,1]
    
    nmae = normalized_mae(y_true.flatten(),y_pred.flatten())
    nmse = normalized_mse(y_true.flatten(),y_pred.flatten())

    
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
    plt.text(0.05, 0.95, #  调整方块的位置
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
    print(f"Normalized MSE: {mse:.4f}, Normalized MAE: {mae:.4f}, R² score: {r2:.4f}, Correlation: {correlation_percentage:.4f}")
    
    return normalized_mse, normalized_mae, r2, correlation_percentage

# Set file paths
afs_train_file = 'Mar1_data\\AFS_train_filled.txt'
theta_train_file = 'Mar1_data\\theta_train.txt'

afs_val_file = 'Mar1_data\\AFS_valid_filled.txt'
theta_val_file = 'Mar1_data\\theta_valid.txt'

afs_test_file = 'Mar1_data\\AFS_test_filled.txt'
theta_test_file = 'Mar1_data\\theta_test.txt'

X_train_raw, y_train_raw = load_and_preprocess_data(afs_train_file, theta_train_file)
X_val_raw, y_val_raw = load_and_preprocess_data(afs_val_file, theta_val_file)
X_test_raw, y_test_raw = load_and_preprocess_data(afs_test_file, theta_test_file)

baseline_results= train_baseline_models(
            X_train_raw, y_train_raw,
            X_val_raw, y_val_raw,
            X_test_raw, y_test_raw
        )

rf_train_result = plot_predictions(y_train_raw, baseline_results['random_forest']['train_pred'], "Random_Forest_Training_Set")
rf_val_result = plot_predictions(y_val_raw, baseline_results['random_forest']['val_pred'], "Random_Forest_Validation_Set")
rf_test_result = plot_predictions(y_test_raw, baseline_results['random_forest']['test_pred'], "Random_Forest_Test_Set")
        