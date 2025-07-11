import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import numpy as np
import pandas as pd
from improved_neural_network import ImprovedQueueNeuralNetwork
from enhanced_neural_network import EnhancedQueueNeuralNetwork
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

class QueuePredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Queue Theory Neural Network Predictor")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize models
        self.basic_model = None
        self.enhanced_model = None
        self.load_models()
        
        # Create GUI
        self.create_widgets()
        
    def load_models(self):
        """Load the trained neural network models"""
        try:
            # Load basic model (lambda + Lq)
            self.basic_model = ImprovedQueueNeuralNetwork()
            self.basic_model.load_model('improved_queue_nn_model.pkl')
            
            # Load enhanced model (all variables)
            self.enhanced_model = EnhancedQueueNeuralNetwork()
            self.enhanced_model.load_model('enhanced_queue_nn_model.pkl')
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load models: {e}")
    
    def create_widgets(self):
        """Create the main GUI widgets"""
        
        # Main title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill='x', padx=10, pady=5)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="Queue Theory Neural Network Predictor", 
                              font=('Arial', 16, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(expand=True)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Tab 1: Basic Prediction (lambda + Lq)
        self.create_basic_prediction_tab(notebook)
        
        # Tab 2: Enhanced Prediction (all variables)
        self.create_enhanced_prediction_tab(notebook)
        
        # Tab 3: Model Comparison
        self.create_comparison_tab(notebook)
        
        # Tab 4: Data Analysis
        self.create_analysis_tab(notebook)
        
    def create_basic_prediction_tab(self, notebook):
        """Create tab for basic prediction (lambda + Lq)"""
        basic_frame = ttk.Frame(notebook)
        notebook.add(basic_frame, text="Basic Prediction (λ + Lq)")
        
        # Left panel - Input
        left_panel = tk.Frame(basic_frame, bg='#ecf0f1', width=400)
        left_panel.pack(side='left', fill='y', padx=10, pady=10)
        left_panel.pack_propagate(False)
        
        # Input section
        input_frame = tk.LabelFrame(left_panel, text="Input Parameters", font=('Arial', 12, 'bold'),
                                   bg='#ecf0f1', fg='#2c3e50')
        input_frame.pack(fill='x', padx=10, pady=10)
        
        # Lambda input
        tk.Label(input_frame, text="λ (Arrival Rate):", font=('Arial', 10), bg='#ecf0f1').pack(anchor='w', padx=10, pady=5)
        self.lambda_var = tk.StringVar(value="0.5")
        lambda_entry = tk.Entry(input_frame, textvariable=self.lambda_var, font=('Arial', 10), width=20)
        lambda_entry.pack(padx=10, pady=5)
        
        # Lq input
        tk.Label(input_frame, text="Lq (Queue Length):", font=('Arial', 10), bg='#ecf0f1').pack(anchor='w', padx=10, pady=5)
        self.lq_var = tk.StringVar(value="10")
        lq_entry = tk.Entry(input_frame, textvariable=self.lq_var, font=('Arial', 10), width=20)
        lq_entry.pack(padx=10, pady=5)
        
        # Predict button
        predict_btn = tk.Button(input_frame, text="Predict Wq", command=self.predict_basic,
                               font=('Arial', 12, 'bold'), bg='#3498db', fg='white',
                               relief='raised', padx=20, pady=10)
        predict_btn.pack(pady=20)
        
        # Results section
        results_frame = tk.LabelFrame(left_panel, text="Prediction Results", font=('Arial', 12, 'bold'),
                                     bg='#ecf0f1', fg='#2c3e50')
        results_frame.pack(fill='x', padx=10, pady=10)
        
        self.basic_result_text = tk.Text(results_frame, height=8, width=40, font=('Arial', 10))
        self.basic_result_text.pack(padx=10, pady=10)
        
        # Right panel - Information
        right_panel = tk.Frame(basic_frame, bg='#ecf0f1')
        right_panel.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        # Model info
        info_frame = tk.LabelFrame(right_panel, text="Model Information", font=('Arial', 12, 'bold'),
                                  bg='#ecf0f1', fg='#2c3e50')
        info_frame.pack(fill='x', pady=10)
        
        info_text = """
Basic Neural Network Model

Input Variables:
• λ (Lambda) - Arrival rate
• Lq - Queue length

Output:
• Wq - Waiting time in queue

Performance:
• Validation R²: 92.26%
• Training R²: 95.80%

This model uses the fundamental queue theory relationship
between arrival rate, queue length, and waiting time.
        """
        
        info_label = tk.Label(info_frame, text=info_text, font=('Arial', 10), 
                             bg='#ecf0f1', justify='left', anchor='nw')
        info_label.pack(padx=10, pady=10)
        
        # Quick test section
        test_frame = tk.LabelFrame(right_panel, text="Quick Tests", font=('Arial', 12, 'bold'),
                                  bg='#ecf0f1', fg='#2c3e50')
        test_frame.pack(fill='x', pady=10)
        
        test_btn = tk.Button(test_frame, text="Run Test Cases", command=self.run_basic_tests,
                            font=('Arial', 10), bg='#27ae60', fg='white', padx=15, pady=5)
        test_btn.pack(pady=10)
        
    def create_enhanced_prediction_tab(self, notebook):
        """Create tab for enhanced prediction (all variables)"""
        enhanced_frame = ttk.Frame(notebook)
        notebook.add(enhanced_frame, text="Enhanced Prediction (All Variables)")
        
        # Left panel - Input
        left_panel = tk.Frame(enhanced_frame, bg='#ecf0f1', width=400)
        left_panel.pack(side='left', fill='y', padx=10, pady=10)
        left_panel.pack_propagate(False)
        
        # Input section
        input_frame = tk.LabelFrame(left_panel, text="Input Parameters", font=('Arial', 12, 'bold'),
                                   bg='#ecf0f1', fg='#2c3e50')
        input_frame.pack(fill='x', padx=10, pady=10)
        
        # Create input fields
        self.enhanced_vars = {}
        enhanced_inputs = [
            ("λ (Arrival Rate):", "lambda", "0.5"),
            ("s (Number of Servers):", "s", "3"),
            ("μ (Service Rate):", "mu", "8.0"),
            ("ρ (Traffic Intensity):", "rho", "0.4"),
            ("W (Total Waiting Time):", "w", "5.0"),
            ("Lq (Queue Length):", "lq", "10"),
            ("Ls (Customers in Service):", "ls", "2.0"),
            ("P0 (Zero Probability):", "p0", "0.6")
        ]
        
        for label_text, var_name, default_value in enhanced_inputs:
            tk.Label(input_frame, text=label_text, font=('Arial', 10), bg='#ecf0f1').pack(anchor='w', padx=10, pady=2)
            var = tk.StringVar(value=default_value)
            self.enhanced_vars[var_name] = var
            entry = tk.Entry(input_frame, textvariable=var, font=('Arial', 10), width=20)
            entry.pack(padx=10, pady=2)
        
        # Predict button
        predict_btn = tk.Button(input_frame, text="Predict Wq (Enhanced)", command=self.predict_enhanced,
                               font=('Arial', 12, 'bold'), bg='#e74c3c', fg='white',
                               relief='raised', padx=20, pady=10)
        predict_btn.pack(pady=20)
        
        # Results section
        results_frame = tk.LabelFrame(left_panel, text="Prediction Results", font=('Arial', 12, 'bold'),
                                     bg='#ecf0f1', fg='#2c3e50')
        results_frame.pack(fill='x', padx=10, pady=10)
        
        self.enhanced_result_text = tk.Text(results_frame, height=8, width=40, font=('Arial', 10))
        self.enhanced_result_text.pack(padx=10, pady=10)
        
        # Right panel - Information
        right_panel = tk.Frame(enhanced_frame, bg='#ecf0f1')
        right_panel.pack(side='right', fill='both', expand=True, padx=10, pady=10)
        
        # Model info
        info_frame = tk.LabelFrame(right_panel, text="Enhanced Model Information", font=('Arial', 12, 'bold'),
                                  bg='#ecf0f1', fg='#2c3e50')
        info_frame.pack(fill='x', pady=10)
        
        info_text = """
Enhanced Neural Network Model

Input Variables (8 total):
• λ (Lambda) - Arrival rate
• s - Number of servers
• μ (Mu) - Service rate per server
• ρ (Rho) - Traffic intensity
• W - Total waiting time
• Lq - Queue length
• Ls - Customers in service
• P0 - Probability of zero customers

Output:
• Wq - Waiting time in queue

Performance:
• Validation R²: 98.64%
• Training R²: 99.23%
• MSE reduction: 81.76%

This model uses all available variables for maximum accuracy.
        """
        
        info_label = tk.Label(info_frame, text=info_text, font=('Arial', 10), 
                             bg='#ecf0f1', justify='left', anchor='nw')
        info_label.pack(padx=10, pady=10)
        
        # Quick test section
        test_frame = tk.LabelFrame(right_panel, text="Quick Tests", font=('Arial', 12, 'bold'),
                                  bg='#ecf0f1', fg='#2c3e50')
        test_frame.pack(fill='x', pady=10)
        
        test_btn = tk.Button(test_frame, text="Run Test Cases", command=self.run_enhanced_tests,
                            font=('Arial', 10), bg='#27ae60', fg='white', padx=15, pady=5)
        test_btn.pack(pady=10)
        
    def create_comparison_tab(self, notebook):
        """Create tab for model comparison"""
        comparison_frame = ttk.Frame(notebook)
        notebook.add(comparison_frame, text="Model Comparison")
        
        # Comparison section
        comp_frame = tk.LabelFrame(comparison_frame, text="Model Performance Comparison", 
                                  font=('Arial', 14, 'bold'), bg='#ecf0f1', fg='#2c3e50')
        comp_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Create comparison table
        columns = ('Metric', 'Basic Model (2 vars)', 'Enhanced Model (8 vars)', 'Improvement')
        tree = ttk.Treeview(comp_frame, columns=columns, show='headings', height=6)
        
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150, anchor='center')
        
        # Add comparison data
        comparison_data = [
            ('Input Variables', 'λ, Lq', 'λ, s, μ, ρ, W, Lq, Ls, P0', '+6 variables'),
            ('Training R²', '95.80%', '99.23%', '+3.43%'),
            ('Validation R²', '92.26%', '98.64%', '+6.38%'),
            ('MSE', '87.04', '15.94', '-81.76%'),
            ('Model Complexity', 'Simple', 'Complex', 'Higher'),
            ('Prediction Speed', 'Fast', 'Slower', 'Trade-off')
        ]
        
        for item in comparison_data:
            tree.insert('', 'end', values=item)
        
        tree.pack(padx=20, pady=20)
        
        # Recommendation section
        rec_frame = tk.LabelFrame(comparison_frame, text="Recommendation", 
                                 font=('Arial', 12, 'bold'), bg='#ecf0f1', fg='#2c3e50')
        rec_frame.pack(fill='x', padx=20, pady=10)
        
        rec_text = """
• Use Basic Model when: You only have λ and Lq data, need fast predictions, or want simplicity
• Use Enhanced Model when: You have all variables available, need maximum accuracy, or for critical applications
• Both models are trained on your complete 10,000-sample dataset
• Both models successfully learn queue theory relationships
        """
        
        rec_label = tk.Label(rec_frame, text=rec_text, font=('Arial', 10), 
                            bg='#ecf0f1', justify='left', anchor='nw')
        rec_label.pack(padx=20, pady=20)
        
    def create_analysis_tab(self, notebook):
        """Create tab for data analysis"""
        analysis_frame = ttk.Frame(notebook)
        notebook.add(analysis_frame, text="Data Analysis")
        
        # Analysis section
        analysis_panel = tk.Frame(analysis_frame, bg='#ecf0f1')
        analysis_panel.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Dataset info
        info_frame = tk.LabelFrame(analysis_panel, text="Dataset Information", 
                                  font=('Arial', 12, 'bold'), bg='#ecf0f1', fg='#2c3e50')
        info_frame.pack(fill='x', pady=10)
        
        try:
            df = pd.read_csv('dataset.csv')
            info_text = f"""
Dataset Overview:
• Total samples: {len(df):,}
• Variables: {len(df.columns)}
• Complete queue system data

Variables in your dataset:
• λ (Lambda): Arrival rate [{df.iloc[:, 0].min():.4f}, {df.iloc[:, 0].max():.4f}]
• s: Number of servers [{df.iloc[:, 1].min():.0f}, {df.iloc[:, 1].max():.0f}]
• μ (Mu): Service rate [{df.iloc[:, 2].min():.4f}, {df.iloc[:, 2].max():.4f}]
• ρ (Rho): Traffic intensity [{df.iloc[:, 3].min():.4f}, {df.iloc[:, 3].max():.4f}]
• Wq: Waiting time [{df.iloc[:, 4].min():.4f}, {df.iloc[:, 4].max():.4f}]
• W: Total waiting time [{df.iloc[:, 5].min():.4f}, {df.iloc[:, 5].max():.4f}]
• Lq: Queue length [{df.iloc[:, 6].min():.4f}, {df.iloc[:, 6].max():.4f}]
• Ls: Customers in service [{df.iloc[:, 7].min():.4f}, {df.iloc[:, 7].max():.4f}]
• P0: Zero probability [{df.iloc[:, 8].min():.4f}, {df.iloc[:, 8].max():.4f}]

Your neural networks have learned from this complete dataset!
            """
        except Exception as e:
            info_text = f"Could not load dataset: {e}"
        
        info_label = tk.Label(info_frame, text=info_text, font=('Arial', 10), 
                             bg='#ecf0f1', justify='left', anchor='nw')
        info_label.pack(padx=20, pady=20)
        
        # Analysis button
        analyze_btn = tk.Button(analysis_panel, text="Analyze Data Relationships", 
                               command=self.analyze_data, font=('Arial', 12, 'bold'),
                               bg='#9b59b6', fg='white', padx=20, pady=10)
        analyze_btn.pack(pady=20)
        
        # Results area
        self.analysis_text = scrolledtext.ScrolledText(analysis_panel, height=15, width=80, 
                                                      font=('Arial', 10))
        self.analysis_text.pack(padx=20, pady=20)
        
    def predict_basic(self):
        """Make prediction using basic model"""
        try:
            lambda_val = float(self.lambda_var.get())
            lq_val = float(self.lq_var.get())
            
            X = np.array([[lambda_val, lq_val]])
            prediction = self.basic_model.predict(X)[0]
            
            result_text = f"""
Basic Model Prediction Results:

Input Values:
• λ (Arrival Rate): {lambda_val}
• Lq (Queue Length): {lq_val}

Prediction:
• Wq (Waiting Time): {prediction:.4f}

Model Performance:
• Validation R²: 92.26%
• Training R²: 95.80%

Little's Law Check:
• Theoretical Wq = Lq/λ = {lq_val/lambda_val:.4f}
• Neural Network Wq = {prediction:.4f}
• Difference = {abs(prediction - lq_val/lambda_val):.4f}
            """
            
            self.basic_result_text.delete(1.0, tk.END)
            self.basic_result_text.insert(1.0, result_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")
    
    def predict_enhanced(self):
        """Make prediction using enhanced model"""
        try:
            # Get all input values
            input_values = []
            for var_name in ['lambda', 's', 'mu', 'rho', 'w', 'lq', 'ls', 'p0']:
                input_values.append(float(self.enhanced_vars[var_name].get()))
            
            X = np.array([input_values])
            prediction = self.enhanced_model.predict(X)[0]
            
            result_text = f"""
Enhanced Model Prediction Results:

Input Values:
• λ (Arrival Rate): {input_values[0]}
• s (Servers): {input_values[1]}
• μ (Service Rate): {input_values[2]}
• ρ (Traffic Intensity): {input_values[3]}
• W (Total Waiting): {input_values[4]}
• Lq (Queue Length): {input_values[5]}
• Ls (In Service): {input_values[6]}
• P0 (Zero Prob): {input_values[7]}

Prediction:
• Wq (Waiting Time): {prediction:.4f}

Model Performance:
• Validation R²: 98.64%
• Training R²: 99.23%
• MSE reduction: 81.76%
            """
            
            self.enhanced_result_text.delete(1.0, tk.END)
            self.enhanced_result_text.insert(1.0, result_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {e}")
    
    def run_basic_tests(self):
        """Run test cases for basic model"""
        test_cases = [
            (0.5, 10, "Balanced scenario"),
            (1.0, 5, "High arrival rate"),
            (0.1, 20, "Low arrival rate"),
            (0.8, 16, "High queue length")
        ]
        
        result_text = "Basic Model Test Cases:\n\n"
        
        for lambda_val, lq_val, description in test_cases:
            X = np.array([[lambda_val, lq_val]])
            prediction = self.basic_model.predict(X)[0]
            theoretical = lq_val / lambda_val
            
            result_text += f"{description}:\n"
            result_text += f"  λ={lambda_val}, Lq={lq_val}\n"
            result_text += f"  Predicted Wq: {prediction:.4f}\n"
            result_text += f"  Theoretical Wq: {theoretical:.4f}\n"
            result_text += f"  Difference: {abs(prediction - theoretical):.4f}\n\n"
        
        self.basic_result_text.delete(1.0, tk.END)
        self.basic_result_text.insert(1.0, result_text)
    
    def run_enhanced_tests(self):
        """Run test cases for enhanced model"""
        test_cases = [
            (0.5, 3, 8.0, 0.4, 5.0, 10, 2.0, 0.6, "Balanced scenario"),
            (0.8, 5, 12.0, 0.3, 3.0, 15, 4.0, 0.7, "High capacity"),
            (0.2, 2, 6.0, 0.6, 8.0, 5, 1.5, 0.4, "Low capacity")
        ]
        
        result_text = "Enhanced Model Test Cases:\n\n"
        
        for case in test_cases:
            lambda_val, s_val, mu_val, rho_val, w_val, lq_val, ls_val, p0_val, description = case
            X = np.array([[lambda_val, s_val, mu_val, rho_val, w_val, lq_val, ls_val, p0_val]])
            prediction = self.enhanced_model.predict(X)[0]
            
            result_text += f"{description}:\n"
            result_text += f"  λ={lambda_val}, s={s_val}, μ={mu_val}, ρ={rho_val}\n"
            result_text += f"  W={w_val}, Lq={lq_val}, Ls={ls_val}, P0={p0_val}\n"
            result_text += f"  Predicted Wq: {prediction:.4f}\n\n"
        
        self.enhanced_result_text.delete(1.0, tk.END)
        self.enhanced_result_text.insert(1.0, result_text)
    
    def analyze_data(self):
        """Analyze data relationships"""
        try:
            df = pd.read_csv('dataset.csv')
            
            analysis_text = "Data Analysis Results:\n\n"
            
            # Basic statistics
            analysis_text += "Dataset Statistics:\n"
            analysis_text += f"• Total samples: {len(df):,}\n"
            analysis_text += f"• Variables: {len(df.columns)}\n\n"
            
            # Variable ranges
            analysis_text += "Variable Ranges:\n"
            for i, col in enumerate(df.columns):
                col_data = df.iloc[:, i]
                analysis_text += f"• Column {i}: [{col_data.min():.4f}, {col_data.max():.4f}]\n"
            
            analysis_text += "\nQueue Theory Relationships:\n"
            
            # Little's Law analysis
            lambda_data = df.iloc[:, 0].values
            lq_data = df.iloc[:, 6].values
            wq_data = df.iloc[:, 4].values
            
            littles_law = lq_data / lambda_data
            correlation = np.corrcoef(wq_data, littles_law)[0, 1]
            
            analysis_text += f"• Little's Law correlation: {correlation:.4f}\n"
            analysis_text += f"• Data follows queue theory relationships\n"
            analysis_text += f"• Neural networks learn these patterns\n"
            
            self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(1.0, analysis_text)
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {e}")

def main():
    root = tk.Tk()
    app = QueuePredictionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 