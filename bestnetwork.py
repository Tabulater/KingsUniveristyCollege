import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import itertools
import time
from tensorflow.keras.callbacks import EarlyStopping

# Enable eager execution
tf.compat.v1.enable_eager_execution()

# Load and prepare the data
def prepare_data():
    # Step 1: Load the dataset
    data = pd.read_csv('dataset/dataset.csv')
    
    # Step 2: Define features and target (using same features as Neural Network.py)
    features = ['lambda', 'Lq', 's', 'mu', 'rho']
    X = data[features]
    y = data['Wq']
    
    # Step 3: Remove invalid data (same as Neural Network.py)
    valid_rows = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_rows]
    y = y[valid_rows]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def create_model(layers_config, activation):
    model = keras.Sequential([
        keras.layers.Input(shape=(5,))  # Input layer with shape matching our 5 features: lambda, Lq, s, mu, rho
    ])
    
    # Hidden layers
    for units in layers_config:
        model.add(keras.layers.Dense(units, activation=activation))
    
    # Output layer (linear activation for regression)
    model.add(keras.layers.Dense(1, activation='linear'))
    
    return model

def test_configurations():
    # Get data
    X_train, X_test, y_train, y_test = prepare_data()
    
    # Define different configurations to test (optimized for speed and accuracy)
    hidden_layers = [
        [64, 32],        # Fast, simple architecture
        [128, 64],       # Moderate complexity
        [256, 128, 64],  # Deeper but still efficient
        [32, 32],        # Very lightweight
        [128, 128],      # Balanced width
        [64, 64, 32]     # Moderate depth with decreasing width
    ]
    
    # Optimized choices for speed and performance
    activations = ['relu', 'elu']  # ReLU and ELU are computationally efficient
    learning_rates = [0.001, 0.01]  # Most common effective learning rates
    optimizers = ['adam', 'nadam']  # Adam and NAdam are generally fastest to converge
    
    best_score = 0
    best_config = None
    results = []
    
    total_combinations = len(hidden_layers) * len(activations) * len(optimizers) * len(learning_rates)
    print(f"Testing {total_combinations} different configurations...")
    
    for layer_config, activation, optimizer, lr in itertools.product(
        hidden_layers, activations, optimizers, learning_rates):
        
        print(f"\nTesting configuration:")
        print(f"Layers: {layer_config}")
        print(f"Activation: {activation}")
        print(f"Optimizer: {optimizer}")
        print(f"Learning rate: {lr}")
        
        # Create and compile model
        model = create_model(layer_config, activation)
        model.compile(optimizer=keras.optimizers.get({'class_name': optimizer, 'config': {'learning_rate': lr}}),
                     loss='mse',  # Mean squared error for regression
                     metrics=['mse', 'mae'])  # Track both MSE and MAE
        
        # Early stopping to prevent unnecessary training
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        # Measure training time
        start_time = time.time()
        
        # Train model with early stopping
        history = model.fit(X_train, y_train,
                          validation_split=0.2,
                          epochs=50,
                          batch_size=64,  # Larger batch size for faster training
                          callbacks=[early_stopping],
                          verbose=0)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Evaluate model
        test_loss, test_mse, test_mae = model.evaluate(X_test, y_test, verbose=0)
        
        # Calculate efficiency score (balancing MAE and speed)
        # Lower MAE is better, so we invert it for the score
        efficiency_score = 1 / ((test_mae + 1e-6) * (training_time + 1e-6))  # Avoid division by zero
        
        results.append({
            'layers': layer_config,
            'activation': activation,
            'optimizer': optimizer,
            'learning_rate': lr,
            'mse': test_mse,
            'mae': test_mae,
            'training_time': training_time,
            'efficiency_score': efficiency_score
        })
        
        # Update best configuration based on efficiency score
        if efficiency_score > best_score:
            best_score = efficiency_score
            best_config = {
                'layers': layer_config,
                'activation': activation,
                'optimizer': optimizer,
                'learning_rate': lr,
                'mse': test_mse,
                'mae': test_mae,
                'training_time': training_time,
                'efficiency_score': efficiency_score
            }
    
    return results, best_config, best_score, model

if __name__ == "__main__":
    print("Starting neural network configuration testing...")
    results, best_config, best_score, best_model = test_configurations()
    
    # Print results sorted by efficiency score
    print("\nTop 5 configurations (sorted by efficiency score):")
    sorted_results = sorted(results, key=lambda x: x['efficiency_score'], reverse=True)
    for i, result in enumerate(sorted_results[:5], 1):
        print(f"\n{i}. Configuration:")
        print(f"   Layers: {result['layers']}")
        print(f"   Activation: {result['activation']}")
        print(f"   Optimizer: {result['optimizer']}")
        print(f"   Learning Rate: {result['learning_rate']}")
        print(f"   Mean Squared Error: {result['mse']:.4f}")
        print(f"   Mean Absolute Error: {result['mae']:.4f}")
        print(f"   Training Time: {result['training_time']:.2f} seconds")
        print(f"   Efficiency Score: {result['efficiency_score']:.4f}")
    
    print("\nBest configuration found (best efficiency):")
    print(f"Hidden layers: {best_config['layers']}")
    print(f"Activation function: {best_config['activation']}")
    print(f"Optimizer: {best_config['optimizer']}")
    print(f"Learning Rate: {best_config['learning_rate']}")
    print(f"Mean Squared Error: {best_config['mse']:.4f}")
    print(f"Mean Absolute Error: {best_config['mae']:.4f}")
    print(f"Training time: {best_config['training_time']:.2f} seconds")
    print(f"Efficiency score: {best_config['efficiency_score']:.4f}")
    
    # Save the best model and configuration
    best_model.save('best_model.keras')  # Save in Keras format
    import json
    with open('best_config.json', 'w') as f:
        json.dump(best_config, f, indent=4)