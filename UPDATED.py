import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')
import logging
import argparse
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper functions
from sklearn.base import clone

def evaluate_with_kfold(model_class, X, y, k=5, model_kwargs=None):
    if model_kwargs is None:
        model_kwargs = {}
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    maes, mses, rmses = [], [], []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        model = model_class(**model_kwargs) if callable(model_class) else clone(model_class)
        if hasattr(model, 'train'):
            model.train(X_train, y_train, verbose=False)
            y_pred = model.predict(X_val)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
        maes.append(mean_absolute_error(y_val, y_pred))
        mses.append(mean_squared_error(y_val, y_pred))
        rmses.append(np.sqrt(mses[-1]))
    return np.mean(maes), np.std(maes), np.mean(mses), np.std(mses), np.mean(rmses), np.std(rmses)

def plot_best_fit(x, y_real, y_pred, xlabel, ylabel, title, label_real, label_pred):
    z_real = np.polyfit(x, y_real, 1)
    p_real = np.poly1d(z_real)
    z_pred = np.polyfit(x, y_pred, 1)
    p_pred = np.poly1d(z_pred)
    x_range = np.linspace(x.min(), x.max(), 200)
    plt.figure(figsize=(8,5))
    plt.plot(x_range, p_real(x_range), color='red', label=label_real)
    plt.plot(x_range, p_pred(x_range), color='blue', label=label_pred)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

class ImprovedQueueNeuralNetwork:
    """
    Improved Neural Network for predicting Wq from lambda and Lq
    Learns the actual relationship in the data without assuming Little's Law
    """
    
    def __init__(self, hidden_layers=(256, 128, 64), learning_rate=0.001, max_iter=1000):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.model = None
        self.scaler_X = MinMaxScaler()  # Use MinMaxScaler for better performance
        self.scaler_y = MinMaxScaler()
        
    def build_model(self):
        """Build the neural network using MLPRegressor"""
        self.model = MLPRegressor( 
            hidden_layer_sizes=self.hidden_layers,
            learning_rate_init=self.learning_rate,
            max_iter=self.max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=50,
            verbose=True,
            alpha=0.001,  # L2 regularization
            activation='relu'
        )
        return self.model
    
    def prepare_data(self, X, y):
        """Prepare and scale the data"""
        # Scale features to [0, 1] range
        X_scaled = self.scaler_X.fit_transform(X)
        
        # Scale target to [0, 1] range
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        return X_scaled, y_scaled
    
    def train(self, X, y, validation_split=0.2, verbose=True):
        """Train the neural network"""
        if self.model is None:
            self.build_model()
        
        # Prepare data
        X_scaled, y_scaled = self.prepare_data(X, y)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_scaled, test_size=validation_split, random_state=42
        )
        
        if verbose:
            print(f"Training with {len(X_train)} samples, validating with {len(X_val)} samples")
            print(f"Input features range: X_min={X.min(axis=0)}, X_max={X.max(axis=0)}")
            print(f"Target range: y_min={y.min():.4f}, y_max={y.max():.4f}")
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Calculate training metrics
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        
        # Convert back to original scale for metrics
        y_train_orig = self.scaler_y.inverse_transform(y_train_pred.reshape(-1, 1)).flatten()
        y_val_orig = self.scaler_y.inverse_transform(y_val_pred.reshape(-1, 1)).flatten()
        y_train_true = self.scaler_y.inverse_transform(y_train.reshape(-1, 1)).flatten()
        y_val_true = self.scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
        
        train_mse = mean_squared_error(y_train_true, y_train_orig)
        val_mse = mean_squared_error(y_val_true, y_val_orig)
        train_mae = mean_absolute_error(y_train_true, y_train_orig)
        val_mae = mean_absolute_error(y_val_true, y_val_orig)
        train_rmse = np.sqrt(train_mse)
        val_rmse = np.sqrt(val_mse)
        
        if verbose:
            print(f"Training MSE: {train_mse:.6f}")
            print(f"Validation MSE: {val_mse:.6f}")
            print(f"Training MAE: {train_mae:.6f}")
            print(f"Validation MAE: {val_mae:.6f}")
            print(f"Training RMSE: {train_rmse:.6f}")
            print(f"Validation RMSE: {val_rmse:.6f}")
            print(f"Training completed in {self.model.n_iter_} iterations")
        
        return {
            'train_mse': train_mse,
            'val_mse': val_mse,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'train_rmse': train_rmse,
            'val_rmse': val_rmse,
            'n_iterations': self.model.n_iter_
        }
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        X_scaled = self.scaler_X.transform(X)
        y_pred_scaled = self.model.predict(X_scaled)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1))
        
        return y_pred.flatten()
    
    def evaluate(self, X, y_true):
        """Evaluate model performance"""
        y_pred = self.predict(X)
        
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        return {
            'MSE': mse,
            'MAE': mae,
            'RMSE': rmse
        }
    
    def plot_predictions(self, X, y_true, sample_size=1000):
        """Plot actual vs predicted values"""
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
            y_true_sample = y_true[indices]
        else:
            X_sample = X
            y_true_sample = y_true
        
        y_pred = self.predict(X_sample)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true_sample, y_pred, alpha=0.5)
        plt.plot([y_true_sample.min(), y_true_sample.max()], 
                [y_true_sample.min(), y_true_sample.max()], 'r--', lw=2)
        plt.xlabel('Actual Wq')
        plt.ylabel('Predicted Wq')
        plt.title('Actual vs Predicted Wq Values')
        plt.grid(True)
        plt.show()
        
        # Print correlation
        correlation = np.corrcoef(y_true_sample, y_pred)[0, 1]
        print(f"Correlation between actual and predicted: {correlation:.4f}")
    
    def save_model(self, filename='improved_queue_nn_model.pkl', extra_metadata=None):
        """Save the trained model with metadata"""
        model_data = {
            'model': self.model,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'metadata': {
                'trained_on': datetime.now().isoformat(),
                'version': '1.0.0',
                'extra': extra_metadata or {}
            }
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        logger.info(f"Model saved to {filename}")
    
    def load_model(self, filename='improved_queue_nn_model.pkl'):
        """Load a trained model"""
        import pickle
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.scaler_X = model_data['scaler_X']
        self.scaler_y = model_data['scaler_y']
        print(f"Model loaded from '{filename}'")

# Helper for best-fit line plotting
def plot_best_fit_subplot(ax, x, y_real, y_pred, xlabel, ylabel, title):
    z_real = np.polyfit(x, y_real, 1)
    p_real = np.poly1d(z_real)
    z_pred = np.polyfit(x, y_pred, 1)
    p_pred = np.poly1d(z_pred)
    x_range = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_range, p_real(x_range), color='red', label='Real Wq (Best Fit)')
    ax.plot(x_range, p_pred(x_range), color='blue', label='Predicted Wq (Best Fit)')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

def train_and_evaluate_nn(X, y, hidden_layers, kfolds, use_kfold, logger, label):
    X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    kfold_metrics = None
    if use_kfold:
        mlp_args = {'hidden_layers': hidden_layers}
        mlp_mae, mlp_mae_std, mlp_mse, mlp_mse_std, mlp_rmse, mlp_rmse_std = evaluate_with_kfold(
            ImprovedQueueNeuralNetwork, X_trainval, y_trainval, k=kfolds, model_kwargs=mlp_args)
        logger.info(f"MLP KFold CV ({label}): MAE={mlp_mae:.4f}±{mlp_mae_std:.4f}, MSE={mlp_mse:.4f}±{mlp_mse_std:.4f}, RMSE={mlp_rmse:.4f}±{mlp_rmse_std:.4f}")
        kfold_metrics = (mlp_mae, mlp_mae_std, mlp_mse, mlp_mse_std, mlp_rmse, mlp_rmse_std)
    nn = ImprovedQueueNeuralNetwork(hidden_layers=hidden_layers)
    nn.train(X_trainval, y_trainval, verbose=True)
    y_pred_full = nn.predict(X)
    y_test_pred = nn.predict(X_test)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    logger.info(f"MLP Hold-out Test ({label}): MAE={test_mae:.4f}, MSE={test_mse:.4f}, RMSE={test_rmse:.4f}")
    return nn, y_pred_full, (test_mae, test_mse, test_rmse), kfold_metrics

def train_improved_model(hidden_layers=(32,), kfolds=5, verbosity=logging.INFO, use_kfold=True):
    logger.setLevel(verbosity)
    datasets = [
        ("dataset/MM1.csv", "MM1"),
        ("dataset/MMS.csv", "MMS")
    ]
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plot_titles = ['Wq vs Lambda', 'Wq vs Lq', 'Wq vs Rho']
    summary = []
    trained_models = {}
    for row, (file_path, label) in enumerate(datasets):
        logger.info(f"\n{'='*30}\nTraining on {label} ({file_path})\n{'='*30}")
        df = pd.read_csv(file_path)
        if label == "MM1":
            lambda_col, lq_col, wq_col = 'λ', 'Lq', 'Wq'
            X = df[[lambda_col, lq_col]].values
            y = df[wq_col].values
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) | np.isinf(X).any(axis=1) | np.isinf(y))
            X, y = X[mask], y[mask]
            lambda_vals, lq_vals, wq_real = X[:, 0], X[:, 1], y
            nn, y_pred, test_metrics, kfold_metrics = train_and_evaluate_nn(X, y, hidden_layers, kfolds, use_kfold, logger, label)
            trained_models[label] = nn
            summary.append((label, test_metrics, kfold_metrics))
            plot_best_fit_subplot(axes[row, 0], lambda_vals, wq_real, y_pred, 'Lambda (λ)', 'Wq', f'{label}: {plot_titles[0]}')
            plot_best_fit_subplot(axes[row, 1], lq_vals, wq_real, y_pred, 'Lq', 'Wq', f'{label}: {plot_titles[1]}')
            axes[row, 2].axis('off')
            axes[row, 2].set_title(f'{label}: {plot_titles[2]} (N/A)')
        elif label == "MMS":
            lambda_col, lq_col, s_col, mu_col, rho_col, wq_col = 'λ', 'Lq', 's', 'μ', 'ρ', 'Wq'
            feature_cols = [lambda_col, lq_col, s_col, mu_col, rho_col]
            X = df[feature_cols].values
            y = df[wq_col].values
            mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) | np.isinf(X).any(axis=1) | np.isinf(y))
            X, y = X[mask], y[mask]
            lambda_vals, lq_vals, rho_vals, wq_real = X[:, 0], X[:, 1], X[:, 4], y
            nn, y_pred, test_metrics, kfold_metrics = train_and_evaluate_nn(X, y, hidden_layers, kfolds, use_kfold, logger, label)
            trained_models[label] = nn
            summary.append((label, test_metrics, kfold_metrics))
            plot_best_fit_subplot(axes[row, 0], lambda_vals, wq_real, y_pred, 'Lambda (λ)', 'Wq', f'{label}: {plot_titles[0]}')
            plot_best_fit_subplot(axes[row, 1], lq_vals, wq_real, y_pred, 'Lq', 'Wq', f'{label}: {plot_titles[1]}')
            plot_best_fit_subplot(axes[row, 2], rho_vals, wq_real, y_pred, 'Rho (ρ)', 'Wq', f'{label}: {plot_titles[2]}')
    plt.tight_layout()
    plt.show()
    # Print summary table
    print("\n==================== MODEL PERFORMANCE SUMMARY ====================")
    print(f"{'Dataset':<8} | {'MAE':>10} | {'MSE':>10} | {'RMSE':>10} | KFold MAE (±std)")
    print("-"*65)
    for label, (mae, mse, rmse), kfold in summary:
        kfold_str = f"{kfold[0]:.4f} (±{kfold[1]:.4f})" if kfold else "-"
        print(f"{label:<8} | {mae:10.4f} | {mse:10.4f} | {rmse:10.4f} | {kfold_str}")
    print("==================================================================\n")
    # Interactive prediction
    print("\n--- INTERACTIVE PREDICTION ---")
    while True:
        print("\nChoose dataset to predict with: 1) MM1  2) MMS  (or type 'exit' to quit)")
        choice = input("Enter 1 or 2: ").strip()
        if choice.lower() == 'exit':
            break
        if choice == '1':
            nn = trained_models['MM1']
            try:
                lam = float(input("Enter lambda (λ): "))
                lq = float(input("Enter Lq: "))
                X_input = np.array([[lam, lq]])
                wq_pred = nn.predict(X_input)[0]
                print(f"Predicted Wq (MM1): {wq_pred:.4f}")
            except Exception as e:
                print(f"Invalid input: {e}")
        elif choice == '2':
            nn = trained_models['MMS']
            try:
                lam = float(input("Enter lambda (λ): "))
                lq = float(input("Enter Lq: "))
                s = float(input("Enter s: "))
                mu = float(input("Enter mu (μ): "))
                rho = float(input("Enter rho (ρ): "))
                X_input = np.array([[lam, lq, s, mu, rho]])
                wq_pred = nn.predict(X_input)[0]
                print(f"Predicted Wq (MMS): {wq_pred:.4f}")
            except Exception as e:
                print(f"Invalid input: {e}")
        else:
            print("Invalid choice. Please enter 1, 2, or 'exit'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate queue neural network models.")
    parser.add_argument('--hidden_layers', type=int, nargs='+', default=[32], help='Hidden layer sizes for MLP')
    parser.add_argument('--kfolds', type=int, default=5, help='Number of KFold splits')
    parser.add_argument('--no_kfold', action='store_true', help='Skip KFold cross-validation for quick test')
    parser.add_argument('--verbosity', type=str, default='INFO', help='Logging verbosity (DEBUG, INFO, WARNING, ERROR)')
    args = parser.parse_args()
    level = getattr(logging, args.verbosity.upper(), logging.INFO)
    train_improved_model(hidden_layers=tuple(args.hidden_layers), kfolds=args.kfolds, verbosity=level, use_kfold=not args.no_kfold) 