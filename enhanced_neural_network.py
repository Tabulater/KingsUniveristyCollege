import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

class EnhancedQueueNeuralNetwork:
    """
    Enhanced Neural Network for predicting Wq using ALL available variables
    Uses: lambda, s, μ, ρ, W, Lq, Ls, P0
    """
    
    def __init__(self, hidden_layers=(512, 256, 128, 64), learning_rate=0.001, max_iter=1000):
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        
    def build_model(self):
        """Build the enhanced neural network"""
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            learning_rate_init=self.learning_rate,
            max_iter=self.max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=50,
            verbose=True,
            alpha=0.001,
            activation='relu'
        )
        return self.model
    
    def prepare_data(self, X, y):
        """Prepare and scale the data"""
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        return X_scaled, y_scaled
    
    def train(self, X, y, validation_split=0.2, verbose=True):
        """Train the enhanced neural network"""
        if self.model is None:
            self.build_model()
        
        # Prepare data
        X_scaled, y_scaled = self.prepare_data(X, y)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_scaled, test_size=validation_split, random_state=42
        )
        
        if verbose:
            print(f"Training enhanced neural network with {len(X_train)} samples")
            print(f"Input features: {X.shape[1]} variables")
            print(f"Feature ranges: X_min={X.min(axis=0)}, X_max={X.max(axis=0)}")
            print(f"Target range: y_min={y.min():.4f}, y_max={y.max():.4f}")
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Calculate metrics
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)
        
        # Convert back to original scale
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
    
    def save_model(self, filename='enhanced_queue_nn_model.pkl'):
        """Save the trained model"""
        import pickle
        model_data = {
            'model': self.model,
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y
        }
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Enhanced model saved to {filename}")
    
    def load_model(self, filename='enhanced_queue_nn_model.pkl'):
        """Load a trained model"""
        import pickle
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        self.model = model_data['model']
        self.scaler_X = model_data['scaler_X']
        self.scaler_y = model_data['scaler_y']
        print(f"Enhanced model loaded from '{filename}'")

def train_enhanced_model():
    """Train the enhanced neural network with all variables"""
    
    # Load data
    df = pd.read_csv('dataset.csv')
    
    # Define all variables based on our analysis
    variables = {
        'lambda': df.columns[0],    # Arrival rate
        's': df.columns[1],         # Number of servers
        'mu': df.columns[2],        # Service rate per server
        'rho': df.columns[3],       # Traffic intensity
        'Wq': df.columns[4],        # Waiting time in queue (target)
        'W': df.columns[5],         # Total waiting time
        'Lq': df.columns[6],        # Queue length
        'Ls': df.columns[7],        # Customers in service
        'P0': df.columns[8]         # Probability of zero customers
    }
    
    print("="*70)
    print("ENHANCED NEURAL NETWORK TRAINING")
    print("Using ALL available variables for better Wq prediction")
    print("="*70)
    
    # Prepare input features (all except Wq which is our target)
    input_features = [variables['lambda'], variables['s'], variables['mu'], 
                     variables['rho'], variables['W'], variables['Lq'], 
                     variables['Ls'], variables['P0']]
    
    X = df[input_features].values
    y = df[variables['Wq']].values
    
    # Remove NaNs and infinite values
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y) | np.isinf(X).any(axis=1) | np.isinf(y))
    X = X[mask]
    y = y[mask]
    
    print(f"Training enhanced neural network with {len(X)} samples")
    print(f"Using {X.shape[1]} input features:")
    for i, feature in enumerate(input_features):
        print(f"  {i+1}. {feature}")
    
    # Train enhanced model
    enhanced_nn = EnhancedQueueNeuralNetwork()
    results = enhanced_nn.train(X, y, verbose=True)

    # Save enhanced model
    enhanced_nn.save_model('enhanced_queue_nn_model.pkl')

    # --- PLOTTING SECTION ---
    # Reload data for plotting
    lambda_vals = df[variables['lambda']].values[mask]
    lq_vals = df[variables['Lq']].values[mask]
    rho_vals = df[variables['rho']].values[mask]
    wq_real = y
    # For enhanced model, X uses all features, but for plotting we want to show predictions as function of lambda, Lq, rho
    # We'll use the same X as used for training
    wq_pred = enhanced_nn.predict(X)

    # Plot Wq vs lambda
    plt.figure(figsize=(8,5))
    plt.scatter(lambda_vals, wq_real, color='red', label='Real Wq', alpha=0.6)
    plt.scatter(lambda_vals, wq_pred, color='blue', label='Predicted Wq', alpha=0.6)
    plt.xlabel('Lambda (λ)')
    plt.ylabel('Wq')
    plt.title('Wq vs Lambda')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Wq vs Lq
    plt.figure(figsize=(8,5))
    plt.scatter(lq_vals, wq_real, color='red', label='Real Wq', alpha=0.6)
    plt.scatter(lq_vals, wq_pred, color='blue', label='Predicted Wq', alpha=0.6)
    plt.xlabel('Lq')
    plt.ylabel('Wq')
    plt.title('Wq vs Lq')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Wq vs Rho
    plt.figure(figsize=(8,5))
    plt.scatter(rho_vals, wq_real, color='red', label='Real Wq', alpha=0.6)
    plt.scatter(rho_vals, wq_pred, color='blue', label='Predicted Wq', alpha=0.6)
    plt.xlabel('Rho (ρ)')
    plt.ylabel('Wq')
    plt.title('Wq vs Rho')
    plt.legend()
    plt.grid(True)
    plt.show()
    # --- END PLOTTING SECTION ---
    
    # Compare with original model
    print(f"\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    
    # Load original model for comparison
    from improved_neural_network import ImprovedQueueNeuralNetwork
    original_nn = ImprovedQueueNeuralNetwork()
    try:
        original_nn.load_model('improved_queue_nn_model.pkl')
        
        # Test original model on same data
        X_original = df[[variables['lambda'], variables['Lq']]].values
        X_original = X_original[mask]
        y_original = y[mask]
        
        y_pred_original = original_nn.predict(X_original)
        original_mse = mean_squared_error(y_original, y_pred_original)
        original_mae = mean_absolute_error(y_original, y_pred_original)
        original_rmse = np.sqrt(original_mse)
        
        # Test enhanced model
        y_pred_enhanced = enhanced_nn.predict(X)
        enhanced_mse = mean_squared_error(y, y_pred_enhanced)
        enhanced_mae = mean_absolute_error(y, y_pred_enhanced)
        enhanced_rmse = np.sqrt(enhanced_mse)
        
        print(f"Original Model (2 features):")
        print(f"  MSE: {original_mse:.4f}")
        print(f"  MAE: {original_mae:.4f}")
        print(f"  RMSE: {original_rmse:.4f}")
        
        print(f"\nEnhanced Model (8 features):")
        print(f"  MSE: {enhanced_mse:.4f}")
        print(f"  MAE: {enhanced_mae:.4f}")
        print(f"  RMSE: {enhanced_rmse:.4f}")
        
        improvement_mse = (original_mse - enhanced_mse) / original_mse * 100
        improvement_mae = (original_mae - enhanced_mae) / original_mae * 100
        improvement_rmse = (original_rmse - enhanced_rmse) / original_rmse * 100
        
        print(f"\nImprovement:")
        print(f"  MSE reduction: {improvement_mse:.2f}%")
        print(f"  MAE reduction: {improvement_mae:.2f}%")
        print(f"  RMSE reduction: {improvement_rmse:.2f}%")
        
        if enhanced_mse < original_mse:
            print(f"\n✅ Enhanced model performs better!")
        else:
            print(f"\n⚠️  Enhanced model doesn't show significant improvement")
            
    except Exception as e:
        print(f"Could not compare with original model: {e}")
    
    # Test predictions
    print(f"\n" + "="*70)
    print("ENHANCED MODEL PREDICTIONS")
    print("="*70)
    
    test_cases = [
        (0.5, 3, 8.0, 0.4, 5.0, 10, 2.0, 0.6, "Balanced scenario"),
        (0.8, 5, 12.0, 0.3, 3.0, 15, 4.0, 0.7, "High capacity scenario"),
        (0.2, 2, 6.0, 0.6, 8.0, 5, 1.5, 0.4, "Low capacity scenario")
    ]
    
    for i, (lambda_val, s_val, mu_val, rho_val, w_val, lq_val, ls_val, p0_val, description) in enumerate(test_cases):
        X_test = np.array([[lambda_val, s_val, mu_val, rho_val, w_val, lq_val, ls_val, p0_val]])
        pred = enhanced_nn.predict(X_test)[0]
        print(f"Test {i+1} ({description}):")
        print(f"  Input: λ={lambda_val}, s={s_val}, μ={mu_val}, ρ={rho_val}, W={w_val}, Lq={lq_val}, Ls={ls_val}, P0={p0_val}")
        print(f"  Predicted Wq: {pred:.4f}")
        print()
    
    return enhanced_nn

if __name__ == "__main__":
    train_enhanced_model() 