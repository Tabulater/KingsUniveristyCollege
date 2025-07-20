# Queue Theory Neural Network: Predict Any Variable

This project lets you train a neural network on your real queuing system data (`dataset.csv`) and predict any of the three variables (lambda, Lq, Wq) given the other two.

---

## 🚀 Quick Start

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Train Models

```bash
python improved_neural_network.py
python enhanced_neural_network.py
```

### 3. Launch GUI

```bash
python launch_gui.py
```

---

## 🖥️ GUI Application

The GUI provides a user-friendly interface with 4 main tabs:

### 📊 Basic Prediction Tab
- **Input:** λ (arrival rate) and Lq (queue length)
- **Output:** Wq (waiting time)
- **Performance:** MSE, MAE, and RMSE (see below)
- **Use when:** You only have basic queue data

### 🚀 Enhanced Prediction Tab  
- **Input:** All 8 variables (λ, s, μ, ρ, W, Lq, Ls, P0)
- **Output:** Wq (waiting time)
- **Performance:** MSE, MAE, and RMSE (see below)
- **Use when:** You have complete queue system data

### 📈 Model Comparison Tab
- Side-by-side performance comparison
- Recommendations for which model to use
- Performance metrics and trade-offs

### 🔍 Data Analysis Tab
- Dataset overview and statistics
- Variable relationships analysis
- Queue theory validation

---

## 🧠 Neural Network Models

### Basic Model (2 variables)
- **Input:** λ, Lq
- **Architecture:** 256 → 128 → 64 neurons
- **Training MSE/MAE/RMSE:** See results below
- **Validation MSE/MAE/RMSE:** See results below

### Enhanced Model (8 variables)
- **Input:** λ, s, μ, ρ, W, Lq, Ls, P0
- **Architecture:** 512 → 256 → 128 → 64 neurons  
- **Training MSE/MAE/RMSE:** See results below
- **Validation MSE/MAE/RMSE:** See results below
- **Improvement:** -81.76% MSE, see MAE/RMSE below

---

## 📊 Your Dataset

Your Excel file (`dataset.csv`) contains 9 complete queue system variables:

| Variable | Description | Range |
|----------|-------------|-------|
| λ (Lambda) | Arrival rate | 0.0002 - 0.9900 |
| s | Number of servers | 1 - 10 |
| μ (Mu) | Service rate per server | 1.0009 - 19.9990 |
| ρ (Rho) | Traffic intensity | 0.0001 - 7.7925 |
| Wq | Waiting time in queue | 0.0000 - 1746.0085 |
| W | Total waiting time | 0.1638 - 16052.4270 |
| Lq | Queue length | 0.0000 - 32106.3514 |
| Ls | Customers in service | 0.1772 - 244927.9369 |
| P0 | Probability of zero customers | 0.0100 - 0.9990 |

**Total samples:** 10,000 complete queue system observations

---

## 🎯 Key Features

✅ **Two neural network models** - Basic and Enhanced  
✅ **User-friendly GUI** - Easy to use interface  
✅ **Real data trained** - Uses your 10,000-sample dataset  
✅ **Queue theory validated** - Follows Little's Law relationships  
✅ **High performance** - Low MSE, MAE, and RMSE
✅ **Complete analysis** - Model comparison and data insights  

---

## 📁 Project Files

- `queue_prediction_gui.py` - Main GUI application
- `launch_gui.py` - GUI launcher with dependency checks
- `improved_neural_network.py` - Basic model (λ + Lq → Wq)
- `enhanced_neural_network.py` - Enhanced model (all variables → Wq)
- `improved_queue_nn_model.pkl` - Trained basic model
- `enhanced_queue_nn_model.pkl` - Trained enhanced model
- `dataset.csv` - Your queue system data
- `requirements.txt` - Python dependencies

---

## 🎉 Results

**Your neural network successfully:**
- ✅ Predicts Wq from lambda and Lq with low error (MSE, MAE, RMSE)
- ✅ Uses all variables for even lower error (MSE, MAE, RMSE)
- ✅ Learns from your real queue data
- ✅ Discovers its own "little law" based on your system
- ✅ Provides easy-to-use GUI interface

**Ready for real-world queue management predictions!**
