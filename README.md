# Queue Prediction Models

## 🎯 Project Overview

This project implements three machine learning models to predict queue waiting times (Wq) using queue theory features:

- **Neural Network**: TensorFlow/Keras implementation
- **Random Forest**: Scikit-learn ensemble method
- **XGBoost**: Gradient boosting implementation

## 📁 Project Structure

```
Neural Network/
├── Models/
│   ├── Neural Network.py    # Neural network for Wq prediction
│   ├── Random Forest.py     # Random Forest for Wq prediction
│   └── XGBoost.py          # XGBoost for Wq prediction
├── Notebooks/
│   ├── Neural_Network_Detailed.ipynb   # Detailed neural network notebook
│   ├── Random_Forest_Detailed.ipynb    # Detailed random forest notebook
│   └── XGBoost_Detailed.ipynb          # Detailed XGBoost notebook
├── dataset/
│   ├── dataset.csv          # Main dataset with queue metrics
│   ├── MM1.csv             # MM1 queue model data
│   └── MMS.csv             # MMS queue model data
├── best_config.json         # Best neural network hyperparameters
├── bestnetwork.py           # Hyperparameter optimization script
├── descriptive_analysis.py  # Basic data statistics
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```
└── README.md               # Project documentation
## 📁 Project Structure


## 📊 Data

The models use queue simulation data with the following features:
- `lambda`: Arrival rate
- `Lq`: Queue length
- `s`: Number of servers
- `mu`: Service rate
- `rho`: Utilization factor
- `Wq`: Waiting time (target variable)

## 🚀 Getting Started

### Prerequisites

Install required packages:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib xgboost
```

### Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`

## 🔧 Usage

### Neural Network Model

```bash
python "Models/Neural Network.py"
```

Features:
- 2 hidden layers (32, 16 neurons)
- ReLU activation
- Adam optimizer
- MSE loss function
- Performance metrics: MAE, MSE, RMSE

### Random Forest Model

```bash
python "Models/Random Forest.py"
```

Features:
- 100 estimators
- Default hyperparameters
- Performance metrics: MAE, MSE, RMSE

### XGBoost Model

```bash
python "Models/XGBoost.py"
```

Features:
- 100 estimators
- Learning rate: 0.1
- Max depth: 6
- Performance metrics: MAE, MSE, RMSE

### Hyperparameter Optimization

```bash
python bestnetwork.py
```

Tests different neural network configurations to find optimal hyperparameters.

### Data Analysis

```bash
python descriptive_analysis.py
```

Provides basic statistical summary of the dataset.

## 📈 Model Performance

All models output:

- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- Sample predictions for first 10 test cases
- Visualization plots comparing actual vs predicted values

## 🎛️ Configuration

The `best_config.json` contains the optimal neural network configuration found through hyperparameter tuning:

- Layers: [64, 32]
- Activation: ReLU
- Optimizer: Adam
- Learning rate: 0.01

## 📝 Key Features

- **Regression models**: All models predict continuous Wq values
- **Standardized features**: Data preprocessing with StandardScaler
- **Consistent evaluation**: Same metrics across all models
- **Visualization**: Scatter plots for feature vs target relationships

## 👨‍💻 Author

Aashrith Raj Tatipamula

## 🙏 Acknowledgments

- King's University College for the internship opportunity
- Dr. Felipe Rodrigues for his mentorship

---

**Note**: This project is designed for educational purposes in queue prediction and machine learning applications.
python "Models/Random Forest.py"
