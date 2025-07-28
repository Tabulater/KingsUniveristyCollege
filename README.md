# Neural Network Project - Queue Prediction System

This repository contains a comprehensive machine learning project focused on queue prediction using various algorithms including Neural Networks, Random Forest, and XGBoost.

## 📁 Project Structure

```
Neural Network/
├── best_config.json          # Best neural network configuration
├── bestnetwork.py            # Optimized neural network implementation
├── dataset/
│   ├── dataset.csv           # Main dataset
│   ├── MM1.csv              # MM1 queue model data
│   └── MMS.csv              # MMS queue model data
├── descriptive_analysis.py   # Data analysis and visualization
├── Neural Network.py         # Main neural network implementation
├── Random Forest.py          # Random Forest implementation
└── XGBoost.py               # XGBoost implementation
```

## 🎯 Project Overview

This project implements and compares different machine learning algorithms for queue prediction:

- **Neural Networks**: Deep learning approach with configurable architecture
- **Random Forest**: Ensemble learning method using decision trees
- **XGBoost**: Gradient boosting framework for efficient predictions

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.7+ installed with the following packages:

```bash
pip install pandas numpy scikit-learn tensorflow matplotlib seaborn xgboost
```

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Neural-Network
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Data

The project uses queue simulation data stored in the `dataset/` directory:

- `dataset.csv`: Main dataset with queue metrics
- `MM1.csv`: Single-server queue model data
- `MMS.csv`: Multi-server queue model data

## 🔧 Usage

### Neural Network Model

```python
python "Neural Network.py"
```

The neural network implementation includes:
- Configurable architecture
- Hyperparameter optimization
- Model persistence
- Performance evaluation

### Random Forest Model

```python
python "Random Forest.py"
```

Features:
- Ensemble learning approach
- Feature importance analysis
- Cross-validation
- Performance metrics

### XGBoost Model

```python
python "XGBoost.py"
```

Features:
- Gradient boosting implementation
- Hyperparameter tuning
- Early stopping
- Model evaluation

### Descriptive Analysis

```python
python descriptive_analysis.py
```

Provides:
- Data exploration and visualization
- Statistical analysis
- Feature correlation analysis
- Data quality assessment

## 📈 Model Performance

The project includes comprehensive evaluation metrics for each model:

- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification results

## 🎛️ Configuration

The `best_config.json` file contains the optimal hyperparameters for the neural network model, discovered through systematic hyperparameter tuning.

## 📝 Key Features

- **Multi-algorithm comparison**: Compare performance across different ML approaches
- **Hyperparameter optimization**: Automated tuning for optimal performance
- **Data visualization**: Comprehensive analysis and plotting capabilities
- **Model persistence**: Save and load trained models
- **Cross-validation**: Robust evaluation methodology

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

Created as part of the King's Internship program.

## 🙏 Acknowledgments

- King's College London for the internship opportunity
- Open source machine learning community
- Contributors and mentors

---

**Note**: This project is designed for educational and research purposes in queue prediction and machine learning applications.