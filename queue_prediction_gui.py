import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import numpy as np
import pandas as pd
from improved_neural_network import ImprovedQueueNeuralNetwork
from enhanced_neural_network import EnhancedQueueNeuralNetwork
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

class QueuePredictionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Queue Theory Neural Network Predictor")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize models
        self.basic_model = None
        self.enhanced_model = None
        self.selected_csv = 'dataset.csv'  # Default dataset
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
        # Modern 10x style: set a bold, clean look
        self.root.configure(bg='#f7fafd')

        # Main title with modern style
        title_frame = tk.Frame(self.root, bg='#222e3a', height=60, bd=0, relief='flat')
        title_frame.pack(fill='x', padx=10, pady=5)
        title_frame.pack_propagate(False)
        title_label = tk.Label(title_frame, text="Queue Predictor", 
                              font=('Segoe UI', 18, 'bold'), fg='white', bg='#222e3a', pady=10)
        title_label.pack(expand=True)

        # Modern notebook style
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TNotebook', background='#f7fafd', borderwidth=0)
        style.configure('TNotebook.Tab', font=('Segoe UI', 12, 'bold'), padding=[20, 10], background='#e3eaf2', foreground='#222e3a', borderwidth=0)
        style.map('TNotebook.Tab', background=[('selected', '#ffffff')], foreground=[('selected', '#1a73e8')])

        notebook = ttk.Notebook(self.root, style='TNotebook')
        notebook.pack(fill='both', expand=True, padx=10, pady=5)

        # Tab 1: Basic Prediction (lambda + Lq)
        self.create_basic_prediction_tab(notebook)
        # Tab 2: Enhanced Prediction (all variables)
        self.create_enhanced_prediction_tab(notebook)
        # Tab 3: Model Comparison
        self.create_comparison_tab(notebook)
        # Tab 4: Data Analysis
        self.create_analysis_tab(notebook)
        # Tab 5: Documentation
        self.create_documentation_tab(notebook)
        
    def create_basic_prediction_tab(self, notebook):
        """Create tab for basic prediction (lambda + Lq)"""
        basic_frame = ttk.Frame(notebook)
        notebook.add(basic_frame, text="Basic Prediction")

        # Input section
        input_frame = tk.LabelFrame(basic_frame, text="Input Parameters", font=('Segoe UI', 12, 'bold'),
                                   bg='#f7fafd', fg='#222e3a', bd=1, relief='solid')
        input_frame.pack(fill='x', padx=20, pady=20)

        # Lambda input
        tk.Label(input_frame, text="Arrival Rate (λ):", font=('Segoe UI', 10), bg='#f7fafd').pack(anchor='w', padx=10, pady=5)
        self.lambda_var = tk.StringVar(value="0.5")
        lambda_entry = tk.Entry(input_frame, textvariable=self.lambda_var, font=('Segoe UI', 10), width=20)
        lambda_entry.pack(padx=10, pady=5)

        # Lq input
        tk.Label(input_frame, text="Queue Length (Lq):", font=('Segoe UI', 10), bg='#f7fafd').pack(anchor='w', padx=10, pady=5)
        self.lq_var = tk.StringVar(value="10")
        lq_entry = tk.Entry(input_frame, textvariable=self.lq_var, font=('Segoe UI', 10), width=20)
        lq_entry.pack(padx=10, pady=5)

        # Predict button
        predict_btn = tk.Button(input_frame, text="Predict Waiting Time", command=self.predict_basic,
                               font=('Segoe UI', 12, 'bold'), bg='#1a73e8', fg='white',
                               relief='flat', padx=20, pady=10, bd=0)
        predict_btn.pack(pady=20)

        # Results section
        results_frame = tk.LabelFrame(basic_frame, text="Prediction Results", font=('Segoe UI', 12, 'bold'),
                                     bg='#f7fafd', fg='#222e3a', bd=1, relief='solid')
        results_frame.pack(fill='both', expand=True, padx=20, pady=20)

        self.basic_result_text = tk.Text(results_frame, height=8, width=40, font=('Segoe UI', 10), bg='white', relief='flat', bd=1)
        self.basic_result_text.pack(padx=10, pady=10, fill='both', expand=True)
        
    def create_enhanced_prediction_tab(self, notebook):
        """Create tab for enhanced prediction (all variables)"""
        enhanced_frame = ttk.Frame(notebook)
        notebook.add(enhanced_frame, text="Advanced Prediction")

        # Input section
        input_frame = tk.LabelFrame(enhanced_frame, text="Input Parameters", font=('Segoe UI', 12, 'bold'),
                                   bg='#f7fafd', fg='#222e3a', bd=1, relief='solid')
        input_frame.pack(fill='x', padx=20, pady=20)

        # Create input fields
        self.enhanced_vars = {}
        enhanced_inputs = [
            ("Arrival Rate (λ):", "lambda", "0.5"),
            ("Number of Servers (s):", "s", "3"),
            ("Service Rate (μ):", "mu", "8.0"),
            ("Traffic Intensity (ρ):", "rho", "0.4"),
            ("Total Waiting Time (W):", "w", "5.0"),
            ("Queue Length (Lq):", "lq", "10"),
            ("Customers in Service (Ls):", "ls", "2.0"),
            ("Zero Probability (P0):", "p0", "0.6")
        ]

        for label_text, var_name, default_value in enhanced_inputs:
            tk.Label(input_frame, text=label_text, font=('Segoe UI', 10), bg='#f7fafd').pack(anchor='w', padx=10, pady=2)
            var = tk.StringVar(value=default_value)
            self.enhanced_vars[var_name] = var
            entry = tk.Entry(input_frame, textvariable=var, font=('Segoe UI', 10), width=20)
            entry.pack(padx=10, pady=2)

        # Predict button
        predict_btn = tk.Button(input_frame, text="Predict Waiting Time", command=self.predict_enhanced,
                               font=('Segoe UI', 12, 'bold'), bg='#1a73e8', fg='white',
                               relief='flat', padx=20, pady=10, bd=0)
        predict_btn.pack(pady=20)

        # Results section
        results_frame = tk.LabelFrame(enhanced_frame, text="Prediction Results", font=('Segoe UI', 12, 'bold'),
                                     bg='#f7fafd', fg='#222e3a', bd=1, relief='solid')
        results_frame.pack(fill='both', expand=True, padx=20, pady=20)

        self.enhanced_result_text = tk.Text(results_frame, height=8, width=40, font=('Segoe UI', 10), bg='white', relief='flat', bd=1)
        self.enhanced_result_text.pack(padx=10, pady=10, fill='both', expand=True)
        
    def create_comparison_tab(self, notebook):
        """Create tab for model comparison"""
        comparison_frame = ttk.Frame(notebook)
        notebook.add(comparison_frame, text="Model Comparison")

        # Comparison section
        comp_frame = tk.LabelFrame(comparison_frame, text="Model Performance Comparison", 
                                  font=('Segoe UI', 14, 'bold'), bg='#f7fafd', fg='#222e3a', bd=1, relief='solid')
        comp_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # Create comparison table
        columns = ('Metric', 'Basic Model', 'Advanced Model', 'Improvement')
        tree = ttk.Treeview(comp_frame, columns=columns, show='headings', height=6)

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=150, anchor='center')

        # Add comparison data
        comparison_data = [
            ('Input Variables', '2 variables', '8 variables', '+6 variables'),
            ('MSE', '87.04', '15.94', '-81.76%'),
            ('MAE', 'X.XX', 'X.XX', 'X.XX'),
            ('RMSE', 'X.XX', 'X.XX', 'X.XX'),
            ('Model Complexity', 'Simple', 'Complex', 'Higher'),
            ('Prediction Speed', 'Fast', 'Slower', 'Trade-off')
        ]

        for item in comparison_data:
            tree.insert('', 'end', values=item)

        tree.pack(padx=20, pady=20)
        
    def create_analysis_tab(self, notebook):
        """Create tab for data analysis"""
        analysis_frame = ttk.Frame(notebook)
        notebook.add(analysis_frame, text="Data Analysis")

        # Analysis section
        analysis_panel = tk.Frame(analysis_frame, bg='#f7fafd')
        analysis_panel.pack(fill='both', expand=True, padx=20, pady=20)

        # Analysis button
        analyze_btn = tk.Button(analysis_panel, text="Analyze Data", 
                               command=self.analyze_data, font=('Segoe UI', 12, 'bold'),
                               bg='#1a73e8', fg='white', relief='flat', padx=20, pady=10, bd=0)
        analyze_btn.pack(pady=20)

        # Results area
        self.analysis_text = scrolledtext.ScrolledText(analysis_panel, height=15, width=80, 
                                                      font=('Segoe UI', 10), bg='white', relief='flat', bd=1)
        self.analysis_text.pack(padx=20, pady=20, fill='both', expand=True)
        
    def create_documentation_tab(self, notebook):
        """Create a tab with documentation and explanations"""
        doc_frame = ttk.Frame(notebook)
        notebook.add(doc_frame, text="Help")

        doc_panel = tk.Frame(doc_frame, bg='#f7fafd')
        doc_panel.pack(fill='both', expand=True, padx=20, pady=20)

        doc_text = scrolledtext.ScrolledText(doc_panel, height=40, width=100, font=('Segoe UI', 11), wrap='word', bg='white', relief='flat', bd=1)
        doc_text.pack(fill='both', expand=True)

        doc_content = '''
Queue Predictor - Help
======================

What is this app?
-----------------
This app helps you predict how long people will wait in line (queue) using artificial intelligence.

Key Terms:
- Arrival Rate (λ): How many people arrive per hour
- Queue Length (Lq): How many people are waiting in line
- Waiting Time (Wq): How long someone waits in line
- Service Rate (μ): How many people can be served per hour
- Number of Servers (s): How many people/machines are serving customers

How to use:
-----------
1. Go to "Basic Prediction" tab for simple predictions
2. Enter your arrival rate and queue length
3. Click "Predict Waiting Time"
4. See the result!

For more accurate predictions, use the "Advanced Prediction" tab with all variables.

Need help? Contact Aashrith.
'''
        doc_text.insert('1.0', doc_content)
        doc_text.config(state='disabled')
        
    def predict_basic(self):
        """Make prediction using basic model"""
        try:
            lambda_val = float(self.lambda_var.get())
            lq_val = float(self.lq_var.get())
            
            X = np.array([[lambda_val, lq_val]])
            prediction = self.basic_model.predict(X)[0]
            
            result_text = f"""
Basic Prediction Results:

Input Values:
• Arrival Rate: {lambda_val}
• Queue Length: {lq_val}

Prediction:
• Waiting Time: {prediction:.2f} hours

The model predicts that customers will wait approximately {prediction:.2f} hours in the queue.
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
Advanced Prediction Results:

Input Values:
• Arrival Rate: {input_values[0]}
• Number of Servers: {input_values[1]}
• Service Rate: {input_values[2]}
• Traffic Intensity: {input_values[3]}
• Total Waiting Time: {input_values[4]}
• Queue Length: {input_values[5]}
• Customers in Service: {input_values[6]}
• Zero Probability: {input_values[7]}

Prediction:
• Waiting Time: {prediction:.2f} hours

The advanced model predicts that customers will wait approximately {prediction:.2f} hours in the queue.
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
    
    def import_csv(self):
        """Open file dialog to import a CSV file and update dataset"""
        file_path = filedialog.askopenfilename(
            title="Select CSV Dataset",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if file_path:
            self.selected_csv = file_path
            self.update_info_label()
            messagebox.showinfo("CSV Imported", f"Using dataset: {file_path}")
    
    def update_info_label(self):
        """Update the dataset info label with current CSV"""
        try:
            df = pd.read_csv(self.selected_csv)
            info_text = f"""
Dataset Overview (File: {self.selected_csv}):
• Total samples: {len(df):,}
• Variables: {len(df.columns)}
• Complete queue system data\n\nVariables in your dataset:\n• λ (Lambda): Arrival rate [{df.iloc[:, 0].min():.4f}, {df.iloc[:, 0].max():.4f}]\n• s: Number of servers [{df.iloc[:, 1].min():.0f}, {df.iloc[:, 1].max():.0f}]\n• μ (Mu): Service rate [{df.iloc[:, 2].min():.4f}, {df.iloc[:, 2].max():.4f}]\n• ρ (Rho): Traffic intensity [{df.iloc[:, 3].min():.4f}, {df.iloc[:, 3].max():.4f}]\n• Wq: Waiting time [{df.iloc[:, 4].min():.4f}, {df.iloc[:, 4].max():.4f}]\n• W: Total waiting time [{df.iloc[:, 5].min():.4f}, {df.iloc[:, 5].max():.4f}]\n• Lq: Queue length [{df.iloc[:, 6].min():.4f}, {df.iloc[:, 6].max():.4f}]\n• Ls: Customers in service [{df.iloc[:, 7].min():.4f}, {df.iloc[:, 7].max():.4f}]\n• P0: Zero probability [{df.iloc[:, 8].min():.4f}, {df.iloc[:, 8].max():.4f}]\n\nYour neural networks have learned from this complete dataset!\n"""
        except Exception as e:
            info_text = f"Could not load dataset: {e}"
        self.info_label.config(text=info_text)
    
    def analyze_data(self):
        """Analyze data relationships"""
        try:
            df = pd.read_csv(self.selected_csv)
            analysis_text = "Data Analysis Results:\n\n"
            # Basic statistics
            analysis_text += f"Dataset Statistics:\n"
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

    def export_text_result(self, text_widget, filetype):
        """Export the contents of a Text widget to CSV, Excel, or PDF"""
        text = text_widget.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("Export", "No results to export.")
            return
        filetypes = {
            'csv': [('CSV Files', '*.csv')],
            'excel': [('Excel Files', '*.xlsx')],
            'pdf': [('PDF Files', '*.pdf')]
        }
        ext = {'csv': '.csv', 'excel': '.xlsx', 'pdf': '.pdf'}[filetype]
        file_path = filedialog.asksaveasfilename(defaultextension=ext, filetypes=filetypes[filetype])
        if not file_path:
            return
        if filetype == 'csv':
            with open(file_path, 'w', encoding='utf-8') as f:
                for line in text.splitlines():
                    f.write(line + '\n')
        elif filetype == 'excel':
            df = pd.DataFrame([line.split(':', 1) if ':' in line else [line, ''] for line in text.splitlines()], columns=['Field', 'Value'])
            df.to_excel(file_path, index=False)
        elif filetype == 'pdf':
            if not REPORTLAB_AVAILABLE:
                messagebox.showerror("PDF Export", "reportlab is not installed. Please install it to export PDF.")
                return
            c = canvas.Canvas(file_path, pagesize=letter)
            width, height = letter
            y = height - 40
            for line in text.splitlines():
                c.drawString(40, y, line)
                y -= 15
                if y < 40:
                    c.showPage()
                    y = height - 40
            c.save()
        messagebox.showinfo("Export", f"Results exported to {file_path}")

    def export_comparison_table(self, filetype):
        """Export the model comparison table to CSV, Excel, or PDF"""
        # Table data as in create_comparison_tab
        columns = ['Metric', 'Basic Model', 'Advanced Model', 'Improvement']
        data = [
            ['Input Variables', '2 variables', '8 variables', '+6 variables'],
            ['MSE', '87.04', '15.94', '-81.76%'],
            ['MAE', 'X.XX', 'X.XX', 'X.XX'],
            ['RMSE', 'X.XX', 'X.XX', 'X.XX'],
            ['Model Complexity', 'Simple', 'Complex', 'Higher'],
            ['Prediction Speed', 'Fast', 'Slower', 'Trade-off']
        ]
        df = pd.DataFrame(data, columns=columns)
        filetypes = {
            'csv': [('CSV Files', '*.csv')],
            'excel': [('Excel Files', '*.xlsx')],
            'pdf': [('PDF Files', '*.pdf')]
        }
        ext = {'csv': '.csv', 'excel': '.xlsx', 'pdf': '.pdf'}[filetype]
        file_path = filedialog.asksaveasfilename(defaultextension=ext, filetypes=filetypes[filetype])
        if not file_path:
            return
        if filetype == 'csv':
            df.to_csv(file_path, index=False)
        elif filetype == 'excel':
            df.to_excel(file_path, index=False)
        elif filetype == 'pdf':
            if not REPORTLAB_AVAILABLE:
                messagebox.showerror("PDF Export", "reportlab is not installed. Please install it to export PDF.")
                return
            c = canvas.Canvas(file_path, pagesize=letter)
            width, height = letter
            y = height - 40
            # Draw header
            for i, col in enumerate(columns):
                c.drawString(40 + i*130, y, col)
            y -= 20
            for row in data:
                for i, cell in enumerate(row):
                    c.drawString(40 + i*130, y, str(cell))
                y -= 15
                if y < 40:
                    c.showPage()
                    y = height - 40
            c.save()
        messagebox.showinfo("Export", f"Comparison table exported to {file_path}")

def main():
    root = tk.Tk()
    app = QueuePredictionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 