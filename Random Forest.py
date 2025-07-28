import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load data
data = pd.read_csv('dataset/dataset.csv')
features = ['lambda', 'Lq', 's', 'mu', 'rho']
X = data[features]
y = data['Wq']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Build Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train model
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate metrics
train_mae = mean_absolute_error(y_train, y_train_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
train_rmse = np.sqrt(train_mse)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)

# Print results
print("RANDOM FOREST PERFORMANCE")
print(f"Training: MAE: {train_mae:.4f}, MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}")
print(f"Testing:  MAE: {test_mae:.4f}, MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}")

# Use the first 10 samples from the test set for consistent comparison across all models
comparison_indices = np.arange(10)  # First 10 indices from test set
y_test_comparison = y_test.iloc[comparison_indices]
y_test_pred_comparison = y_test_pred[comparison_indices]

# Show 10 comparison predictions
comparison = pd.DataFrame({
    'Actual_Wq': y_test_comparison,
    'Predicted_Wq': y_test_pred_comparison,
    'Difference': np.abs(y_test_comparison - y_test_pred_comparison)
})
print("\nSample Predictions (First 10 from Test Set):")
print(comparison.to_string(index=False))

# Create 6 plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Convert back to original scale for plotting
X_train_orig = scaler.inverse_transform(X_train)
X_test_orig = scaler.inverse_transform(X_test)

# Create plots
features_plot = ['lambda', 'Lq', 'rho']
indices = [0, 1, 4]

for row, (name, X_data, y_actual, y_pred) in enumerate([
    ('Training', X_train_orig, y_train, y_train_pred),
    ('Testing', X_test_orig, y_test, y_test_pred)
]):
    for col, (feature, idx) in enumerate(zip(features_plot, indices)):
        ax = axes[row, col]
        ax.scatter(X_data[:, idx], y_actual, alpha=0.6, color='blue', label='Actual', s=20)
        ax.scatter(X_data[:, idx], y_pred, alpha=0.6, color='red', label='Predicted', s=20)
        ax.set_xlabel(feature)
        ax.set_ylabel('Wq')
        ax.set_title(f'{name}: Wq vs {feature}')
        ax.legend()
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()