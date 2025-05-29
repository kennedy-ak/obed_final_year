# #!/usr/bin/env python3
# """
# STREAMLIT TRUCK PRODUCTION PREDICTION APP
# Interactive web application for truck production inference
# Run with: streamlit run app.py
# """

# import streamlit as st
# import numpy as np
# import pandas as pd
# import joblib
# import json
# import os
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots
# import warnings
# warnings.filterwarnings('ignore')

# # Required class definitions for model loading
# from sklearn.preprocessing import StandardScaler

# class ExtremeeLearningMachine:
#     def __init__(self, n_hidden_nodes=100, activation='sigmoid', C=1.0):
#         self.n_hidden_nodes = n_hidden_nodes
#         self.activation = activation
#         self.C = C
#         self.input_weights = None
#         self.biases = None
#         self.output_weights = None
#         self.scaler_X = StandardScaler()
#         self.scaler_y = StandardScaler()
        
#     def _activation_function(self, x):
#         if self.activation == 'sigmoid':
#             return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
#         elif self.activation == 'tanh':
#             return np.tanh(x)
#         elif self.activation == 'relu':
#             return np.maximum(0, x)
#         else:
#             return x
    
#     def fit(self, X, y, input_weights=None, biases=None):
#         X_scaled = self.scaler_X.fit_transform(X)
#         y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
#         n_samples, n_features = X_scaled.shape
        
#         if input_weights is not None and biases is not None:
#             self.input_weights = input_weights
#             self.biases = biases
#         else:
#             np.random.seed(42)
#             self.input_weights = np.random.uniform(-1, 1, (n_features, self.n_hidden_nodes))
#             self.biases = np.random.uniform(-1, 1, self.n_hidden_nodes)
        
#         H = np.dot(X_scaled, self.input_weights) + self.biases
#         H = self._activation_function(H)
        
#         try:
#             if self.C == np.inf:
#                 self.output_weights = np.linalg.pinv(H).dot(y_scaled)
#             else:
#                 identity = np.eye(H.shape[1])
#                 self.output_weights = np.linalg.inv(H.T.dot(H) + identity/self.C).dot(H.T).dot(y_scaled)
#         except:
#             self.output_weights = np.linalg.pinv(H).dot(y_scaled)
    
#     def predict(self, X):
#         X_scaled = self.scaler_X.transform(X)
#         H = np.dot(X_scaled, self.input_weights) + self.biases
#         H = self._activation_function(H)
#         y_pred = np.dot(H, self.output_weights)
#         return self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

# class HybridELM:
#     def __init__(self, optimizer_type='ICA', n_hidden_nodes=80, n_agents=50, max_iter=100, seed=None):
#         self.optimizer_type = optimizer_type
#         self.n_hidden_nodes = n_hidden_nodes
#         self.n_agents = n_agents
#         self.max_iter = max_iter
#         self.elm = ExtremeeLearningMachine(n_hidden_nodes=n_hidden_nodes)
#         self.optimizer = None
#         self.X_train = None
#         self.y_train = None
#         self.seed = seed
#         self.optimized_input_weights = None
#         self.optimized_biases = None
        
#     def predict(self, X):
#         return self.elm.predict(X)

# # =============================================================================
# # STREAMLIT APP CONFIGURATION
# # =============================================================================

# st.set_page_config(
#     page_title="Truck Production Predictor",
#     page_icon="🚛",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS
# st.markdown("""
# <style>
#     .main > div {
#         padding-top: 2rem;
#     }
#     .stSelectbox > div > div > select {
#         background-color: #f0f2f6;
#     }
#     .prediction-card {
#         background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
#         padding: 1rem;
#         border-radius: 10px;
#         color: white;
#         text-align: center;
#         margin: 1rem 0;
#     }
#     .metric-card {
#         background: #f8f9fa;
#         padding: 1rem;
#         border-radius: 8px;
#         border-left: 4px solid #007bff;
#         margin: 0.5rem 0;
#     }
# </style>
# """, unsafe_allow_html=True)

# # =============================================================================
# # LOAD MODELS AND METADATA
# # =============================================================================

# @st.cache_data
# def load_models_and_metadata():
#     """Load all models and metadata"""
#     model_dir = 'saved_models'
    
#     if not os.path.exists(model_dir):
#         return None, None, None
    
#     # Load results
#     try:
#         with open(os.path.join(model_dir, 'results.json'), 'r') as f:
#             results = json.load(f)
#     except:
#         results = {}
    
#     # Load feature info
#     try:
#         with open(os.path.join(model_dir, 'feature_info.json'), 'r') as f:
#             feature_info = json.load(f)
#     except:
#         feature_info = {
#             'feature_names': [
#                 'truck_model_encoded', 'nominal_tonnage', 'material_type_encoded',
#                 'fixed_time', 'variable_time', 'number_of_loads', 'cycle_distance'
#             ],
#             'truck_models': ['KOMATSU HD785', 'CAT 777F', 'CAT 785C', 'CAT 777E', 'KOMATSU HD1500'],
#             'material_types': ['Waste', 'High Grade', 'Low Grade']
#         }
    
#     # Load models
#     models = {}
#     for model_name in results.keys():
#         try:
#             model_path = os.path.join(model_dir, f'{model_name}.joblib')
#             if os.path.exists(model_path):
#                 models[model_name] = joblib.load(model_path)
#         except Exception as e:
#             st.warning(f"Could not load {model_name}: {e}")
    
#     return models, results, feature_info

# # =============================================================================
# # HELPER FUNCTIONS
# # =============================================================================

# def create_model_comparison_chart(results):
#     """Create model performance comparison chart"""
#     if not results:
#         return None
    
#     models = list(results.keys())
#     r2_scores = [results[model]['metrics']['R2'] for model in models]
#     mape_scores = [results[model]['metrics']['MAPE'] for model in models]
    
#     fig = make_subplots(
#         rows=1, cols=2,
#         subplot_titles=('R² Score (Higher is Better)', 'MAPE (Lower is Better)'),
#         specs=[[{"secondary_y": False}, {"secondary_y": False}]]
#     )
    
#     # R² scores
#     fig.add_trace(
#         go.Bar(
#             x=models,
#             y=r2_scores,
#             name='R² Score',
#             marker_color='lightblue',
#             text=[f'{score:.4f}' for score in r2_scores],
#             textposition='auto',
#         ),
#         row=1, col=1
#     )
    
#     # MAPE scores
#     fig.add_trace(
#         go.Bar(
#             x=models,
#             y=mape_scores,
#             name='MAPE (%)',
#             marker_color='lightcoral',
#             text=[f'{score:.2f}%' for score in mape_scores],
#             textposition='auto',
#         ),
#         row=1, col=2
#     )
    
#     fig.update_layout(
#         title_text="Model Performance Comparison",
#         showlegend=False,
#         height=400
#     )
    
#     return fig

# def make_prediction(model, truck_model, nominal_tonnage, material_type, 
#                    fixed_time, variable_time, number_of_loads, cycle_distance,
#                    feature_info):
#     """Make prediction with the selected model"""
    
#     # Encode categorical variables
#     truck_model_encoded = feature_info['truck_models'].index(truck_model)
#     material_type_encoded = feature_info['material_types'].index(material_type)
    
#     # Create input array
#     input_array = np.array([[
#         truck_model_encoded,
#         nominal_tonnage,
#         material_type_encoded,
#         fixed_time,
#         variable_time,
#         number_of_loads,
#         cycle_distance
#     ]])
    
#     # Make prediction
#     prediction = model.predict(input_array)[0]
#     return prediction

# def create_sensitivity_chart(input_params, models, feature_info):
#     """Create sensitivity analysis chart for current inputs"""
#     base_prediction = {}
#     sensitivities = {}
    
#     # Get base predictions for all models
#     for model_name, model in models.items():
#         pred = make_prediction(model, **input_params, feature_info=feature_info)
#         base_prediction[model_name] = pred
    
#     # Calculate sensitivities by varying each parameter
#     param_variations = {
#         'nominal_tonnage': np.linspace(50, 200, 10),
#         'fixed_time': np.linspace(1, 20, 10),
#         'variable_time': np.linspace(0.5, 15, 10),
#         'number_of_loads': np.linspace(1, 100, 10),
#         'cycle_distance': np.linspace(0.1, 20, 10)
#     }
    
#     fig = make_subplots(
#         rows=2, cols=3,
#         subplot_titles=list(param_variations.keys()),
#         specs=[[{"secondary_y": False} for _ in range(3)] for _ in range(2)]
#     )
    
#     positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]
    
#     for i, (param_name, values) in enumerate(param_variations.items()):
#         if i >= len(positions):
#             break
            
#         row, col = positions[i]
        
#         for model_name, model in models.items():
#             predictions = []
#             for val in values:
#                 temp_params = input_params.copy()
#                 temp_params[param_name] = val
#                 pred = make_prediction(model, **temp_params, feature_info=feature_info)
#                 predictions.append(pred)
            
#             fig.add_trace(
#                 go.Scatter(
#                     x=values,
#                     y=predictions,
#                     mode='lines+markers',
#                     name=model_name,
#                     showlegend=(i == 0)
#                 ),
#                 row=row, col=col
#             )
    
#     fig.update_layout(
#         title_text="Parameter Sensitivity Analysis",
#         height=600
#     )
    
#     return fig

# # =============================================================================
# # MAIN APP
# # =============================================================================

# def main():
#     # Load models and metadata
#     models, results, feature_info = load_models_and_metadata()
    
#     if models is None or not models:
#         st.error("❌ No models found! Please run the training script first.")
#         st.info("Make sure the 'saved_models' directory exists with trained models.")
#         return
    
#     # Header
#     st.title("🚛 Truck Production Prediction System")
#     st.markdown("### Interactive prediction tool for mining truck production optimization")
    
#     # Sidebar for model selection and info
#     with st.sidebar:
#         st.header("🔧 Model Configuration")
        
#         # Model selection
#         model_names = list(models.keys())
#         selected_model = st.selectbox(
#             "Select Prediction Model:",
#             model_names,
#             index=0
#         )
        
#         if results:
#             st.markdown("### 📊 Model Performance")
#             model_metrics = results[selected_model]['metrics']
            
#             st.markdown(f"""
#             <div class="metric-card">
#                 <strong>R² Score:</strong> {model_metrics['R2']:.4f}<br>
#                 <strong>MAPE:</strong> {model_metrics['MAPE']:.2f}%<br>
#                 <strong>VAF:</strong> {model_metrics.get('VAF', 'N/A')}<br>
#                 <strong>NASH:</strong> {model_metrics.get('NASH', 'N/A')}
#             </div>
#             """, unsafe_allow_html=True)
        
#         st.markdown("### ℹ️ About the Models")
#         model_descriptions = {
#             'Base_ELM': 'Standard Extreme Learning Machine',
#             'ICA_ELM': 'ELM optimized with Imperialist Competitive Algorithm',
#             'HBO_ELM': 'ELM optimized with Heap-Based Optimizer',
#             'MFO_ELM': 'ELM optimized with Moth-Flame Optimization',
#             'BOA_ELM': 'ELM optimized with Butterfly Optimization Algorithm'
#         }
        
#         if selected_model in model_descriptions:
#             st.info(model_descriptions[selected_model])
    
#     # Main content area
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         st.header("🎛️ Input Parameters")
        
#         # Create input form
#         with st.form("prediction_form"):
#             # Row 1
#             row1_col1, row1_col2, row1_col3 = st.columns(3)
            
#             with row1_col1:
#                 truck_model = st.selectbox(
#                     "Truck Model:",
#                     feature_info['truck_models'],
#                     help="Select the truck model for production"
#                 )
            
#             with row1_col2:
#                 nominal_tonnage = st.slider(
#                     "Nominal Tonnage (tonnes):",
#                     min_value=50.0,
#                     max_value=200.0,
#                     value=100.0,
#                     step=5.0,
#                     help="Truck carrying capacity"
#                 )
            
#             with row1_col3:
#                 material_type = st.selectbox(
#                     "Material Type:",
#                     feature_info['material_types'],
#                     help="Type of material being transported"
#                 )
            
#             # Row 2
#             row2_col1, row2_col2 = st.columns(2)
            
#             with row2_col1:
#                 fixed_time = st.slider(
#                     "Fixed Time (hours):",
#                     min_value=1.0,
#                     max_value=20.0,
#                     value=8.0,
#                     step=0.5,
#                     help="Fixed operational time (FH+EH)"
#                 )
            
#             with row2_col2:
#                 variable_time = st.slider(
#                     "Variable Time (hours):",
#                     min_value=0.5,
#                     max_value=15.0,
#                     value=5.0,
#                     step=0.5,
#                     help="Variable operational time (DT+Q+LT)"
#                 )
            
#             # Row 3
#             row3_col1, row3_col2 = st.columns(2)
            
#             with row3_col1:
#                 number_of_loads = st.slider(
#                     "Number of Loads:",
#                     min_value=1,
#                     max_value=100,
#                     value=20,
#                     step=1,
#                     help="Total number of loads to transport"
#                 )
            
#             with row3_col2:
#                 cycle_distance = st.slider(
#                     "Cycle Distance (km):",
#                     min_value=0.1,
#                     max_value=20.0,
#                     value=5.0,
#                     step=0.1,
#                     help="Round-trip distance for each cycle"
#                 )
            
#             # Predict button
#             predict_button = st.form_submit_button(
#                 "🚀 Predict Production",
#                 use_container_width=True
#             )
    
#     with col2:
#         st.header("📈 Prediction Results")
        
#         if predict_button:
#             # Prepare input parameters
#             input_params = {
#                 'truck_model': truck_model,
#                 'nominal_tonnage': nominal_tonnage,
#                 'material_type': material_type,
#                 'fixed_time': fixed_time,
#                 'variable_time': variable_time,
#                 'number_of_loads': number_of_loads,
#                 'cycle_distance': cycle_distance
#             }
            
#             # Make prediction
#             try:
#                 selected_model_obj = models[selected_model]
#                 prediction = make_prediction(
#                     selected_model_obj, 
#                     **input_params, 
#                     feature_info=feature_info
#                 )
                
#                 # Display prediction
#                 st.markdown(f"""
#                 <div class="prediction-card">
#                     <h2>🎯 Predicted Production</h2>
#                     <h1>{prediction:.2f} tonnes</h1>
#                     <p>Using {selected_model}</p>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 # Additional metrics
#                 efficiency = prediction / (fixed_time + variable_time)
#                 tons_per_load = prediction / number_of_loads
                
#                 st.metric("Production Efficiency", f"{efficiency:.2f} tonnes/hour")
#                 st.metric("Average per Load", f"{tons_per_load:.2f} tonnes/load")
                
#                 # Compare with all models
#                 st.subheader("📊 All Model Predictions")
#                 comparison_data = []
                
#                 for model_name, model in models.items():
#                     pred = make_prediction(model, **input_params, feature_info=feature_info)
#                     r2 = results[model_name]['metrics']['R2'] if results else 0
#                     comparison_data.append({
#                         'Model': model_name,
#                         'Prediction (tonnes)': f"{pred:.2f}",
#                         'R² Score': f"{r2:.4f}"
#                     })
                
#                 comparison_df = pd.DataFrame(comparison_data)
#                 st.dataframe(comparison_df, use_container_width=True)
                
#             except Exception as e:
#                 st.error(f"Prediction failed: {e}")
    
#     # Additional analysis sections
#     if results:
#         st.header("📊 Model Performance Analysis")
        
#         # Model comparison chart
#         comparison_chart = create_model_comparison_chart(results)
#         if comparison_chart:
#             st.plotly_chart(comparison_chart, use_container_width=True)
    
#     # Parameter sensitivity analysis
#     if predict_button and models:
#         st.header("🔍 Parameter Sensitivity Analysis")
#         st.markdown("See how production changes when varying each parameter:")
        
#         sensitivity_chart = create_sensitivity_chart(input_params, models, feature_info)
#         st.plotly_chart(sensitivity_chart, use_container_width=True)
    
#     # Additional information
#     with st.expander("ℹ️ How to Use This App"):
#         st.markdown("""
#         ### 🎯 Quick Start Guide
        
#         1. **Select a Model**: Choose from the trained models in the sidebar
#         2. **Set Parameters**: Adjust the operational parameters using the sliders
#         3. **Predict**: Click "Predict Production" to get results
#         4. **Analyze**: Review the sensitivity analysis to understand parameter impacts
        
#         ### 📊 Understanding the Results
        
#         - **Predicted Production**: Main output in tonnes
#         - **Efficiency**: Production per hour of operation
#         - **Per Load Average**: Production efficiency per individual load
#         - **Model Comparison**: See how different models perform on your inputs
        
#         ### 🔧 Model Types
        
#         - **Base_ELM**: Standard implementation
#         - **ICA_ELM**: Optimized with Imperialist Competitive Algorithm
#         - **HBO_ELM**: Optimized with Heap-Based Optimizer  
#         - **MFO_ELM**: Optimized with Moth-Flame Optimization
#         - **BOA_ELM**: Optimized with Butterfly Optimization Algorithm
        
#         ### 📈 Tips for Optimization
        
#         - Use the sensitivity analysis to identify critical parameters
#         - Compare predictions across different models for validation
#         - Experiment with different parameter combinations
#         - Focus on parameters with high sensitivity indices
#         """)

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
STREAMLIT TRUCK PRODUCTION PREDICTION APP
Interactive web application for truck production inference
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Required class definitions for model loading
from sklearn.preprocessing import StandardScaler

class DiverseOptimizer:
    """Base class ensuring diverse optimization results"""
    def __init__(self, n_agents=50, max_iter=100, dim=None, seed=None, algorithm_id=0):
        self.n_agents = n_agents
        self.max_iter = max_iter
        self.dim = dim
        self.seed = seed
        self.algorithm_id = algorithm_id
        self.best_solution = None
        self.best_fitness = float('inf')
        self.convergence_curve = []

class ICA_Optimizer(DiverseOptimizer):
    def __init__(self, n_countries=50, n_imperialists=10, max_iter=100, dim=None, seed=None):
        super().__init__(n_countries, max_iter, dim, seed, algorithm_id=0)
        self.n_imperialists = n_imperialists
        self.n_colonies = n_countries - n_imperialists

class HBO_Optimizer(DiverseOptimizer):
    def __init__(self, n_agents=50, max_iter=100, dim=None, seed=None):
        super().__init__(n_agents, max_iter, dim, seed, algorithm_id=1)

class MFO_Optimizer(DiverseOptimizer):
    def __init__(self, n_agents=50, max_iter=100, dim=None, seed=None):
        super().__init__(n_agents, max_iter, dim, seed, algorithm_id=2)

class BOA_Optimizer(DiverseOptimizer):
    def __init__(self, n_agents=50, max_iter=100, dim=None, seed=None):
        super().__init__(n_agents, max_iter, dim, seed, algorithm_id=3)
        self.c = 0.02
        self.a = 0.15

class ExtremeeLearningMachine:
    def __init__(self, n_hidden_nodes=100, activation='sigmoid', C=1.0):
        self.n_hidden_nodes = n_hidden_nodes
        self.activation = activation
        self.C = C
        self.input_weights = None
        self.biases = None
        self.output_weights = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
    def _activation_function(self, x):
        if self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'relu':
            return np.maximum(0, x)
        else:
            return x
    
    def fit(self, X, y, input_weights=None, biases=None):
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        n_samples, n_features = X_scaled.shape
        
        if input_weights is not None and biases is not None:
            self.input_weights = input_weights
            self.biases = biases
        else:
            np.random.seed(42)
            self.input_weights = np.random.uniform(-1, 1, (n_features, self.n_hidden_nodes))
            self.biases = np.random.uniform(-1, 1, self.n_hidden_nodes)
        
        H = np.dot(X_scaled, self.input_weights) + self.biases
        H = self._activation_function(H)
        
        try:
            if self.C == np.inf:
                self.output_weights = np.linalg.pinv(H).dot(y_scaled)
            else:
                identity = np.eye(H.shape[1])
                self.output_weights = np.linalg.inv(H.T.dot(H) + identity/self.C).dot(H.T).dot(y_scaled)
        except:
            self.output_weights = np.linalg.pinv(H).dot(y_scaled)
    
    def predict(self, X):
        X_scaled = self.scaler_X.transform(X)
        H = np.dot(X_scaled, self.input_weights) + self.biases
        H = self._activation_function(H)
        y_pred = np.dot(H, self.output_weights)
        return self.scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()

class HybridELM:
    def __init__(self, optimizer_type='ICA', n_hidden_nodes=80, n_agents=50, max_iter=100, seed=None):
        self.optimizer_type = optimizer_type
        self.n_hidden_nodes = n_hidden_nodes
        self.n_agents = n_agents
        self.max_iter = max_iter
        self.elm = ExtremeeLearningMachine(n_hidden_nodes=n_hidden_nodes)
        self.optimizer = None
        self.X_train = None
        self.y_train = None
        self.seed = seed
        self.optimized_input_weights = None
        self.optimized_biases = None
        
    def predict(self, X):
        return self.elm.predict(X)

# =============================================================================
# STREAMLIT APP CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Truck Production Predictor",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stSelectbox > div > div > select {
        background-color: #f0f2f6;
    }
    .prediction-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD MODELS AND METADATA
# =============================================================================

@st.cache_data
def load_models_and_metadata():
    """Load all models and metadata"""
    model_dir = 'saved_models'
    
    if not os.path.exists(model_dir):
        return None, None, None
    
    # Load results
    try:
        with open(os.path.join(model_dir, 'results.json'), 'r') as f:
            results = json.load(f)
    except:
        results = {}
    
    # Load feature info
    try:
        with open(os.path.join(model_dir, 'feature_info.json'), 'r') as f:
            feature_info = json.load(f)
    except:
        feature_info = {
            'feature_names': [
                'truck_model_encoded', 'nominal_tonnage', 'material_type_encoded',
                'fixed_time', 'variable_time', 'number_of_loads', 'cycle_distance'
            ],
            'truck_models': ['KOMATSU HD785', 'CAT 777F', 'CAT 785C', 'CAT 777E', 'KOMATSU HD1500'],
            'material_types': ['Waste', 'High Grade', 'Low Grade']
        }
    
    # Load models
    models = {}
    for model_name in results.keys():
        try:
            model_path = os.path.join(model_dir, f'{model_name}.joblib')
            if os.path.exists(model_path):
                models[model_name] = joblib.load(model_path)
        except Exception as e:
            st.warning(f"Could not load {model_name}: {e}")
    
    return models, results, feature_info

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_model_comparison_chart(results):
    """Create model performance comparison chart"""
    if not results:
        return None
    
    models = list(results.keys())
    r2_scores = [results[model]['metrics']['R2'] for model in models]
    mape_scores = [results[model]['metrics']['MAPE'] for model in models]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('R² Score (Higher is Better)', 'MAPE (Lower is Better)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # R² scores
    fig.add_trace(
        go.Bar(
            x=models,
            y=r2_scores,
            name='R² Score',
            marker_color='lightblue',
            text=[f'{score:.4f}' for score in r2_scores],
            textposition='auto',
        ),
        row=1, col=1
    )
    
    # MAPE scores
    fig.add_trace(
        go.Bar(
            x=models,
            y=mape_scores,
            name='MAPE (%)',
            marker_color='lightcoral',
            text=[f'{score:.2f}%' for score in mape_scores],
            textposition='auto',
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="Model Performance Comparison",
        showlegend=False,
        height=400
    )
    
    return fig

def make_prediction(model, truck_model, nominal_tonnage, material_type, 
                   fixed_time, variable_time, number_of_loads, cycle_distance,
                   feature_info):
    """Make prediction with the selected model"""
    
    # Encode categorical variables
    truck_model_encoded = feature_info['truck_models'].index(truck_model)
    material_type_encoded = feature_info['material_types'].index(material_type)
    
    # Create input array
    input_array = np.array([[
        truck_model_encoded,
        nominal_tonnage,
        material_type_encoded,
        fixed_time,
        variable_time,
        number_of_loads,
        cycle_distance
    ]])
    
    # Make prediction
    prediction = model.predict(input_array)[0]
    return prediction

def create_sensitivity_chart(input_params, models, feature_info):
    """Create sensitivity analysis chart for current inputs"""
    base_prediction = {}
    sensitivities = {}
    
    # Get base predictions for all models
    for model_name, model in models.items():
        pred = make_prediction(model, **input_params, feature_info=feature_info)
        base_prediction[model_name] = pred
    
    # Calculate sensitivities by varying each parameter
    param_variations = {
        'nominal_tonnage': np.linspace(50, 200, 10),
        'fixed_time': np.linspace(1, 20, 10),
        'variable_time': np.linspace(0.5, 15, 10),
        'number_of_loads': np.linspace(1, 100, 10),
        'cycle_distance': np.linspace(0.1, 20, 10)
    }
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=list(param_variations.keys()),
        specs=[[{"secondary_y": False} for _ in range(3)] for _ in range(2)]
    )
    
    positions = [(1,1), (1,2), (1,3), (2,1), (2,2)]
    
    for i, (param_name, values) in enumerate(param_variations.items()):
        if i >= len(positions):
            break
            
        row, col = positions[i]
        
        for model_name, model in models.items():
            predictions = []
            for val in values:
                temp_params = input_params.copy()
                temp_params[param_name] = val
                pred = make_prediction(model, **temp_params, feature_info=feature_info)
                predictions.append(pred)
            
            fig.add_trace(
                go.Scatter(
                    x=values,
                    y=predictions,
                    mode='lines+markers',
                    name=model_name,
                    showlegend=(i == 0)
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        title_text="Parameter Sensitivity Analysis",
        height=600
    )
    
    return fig

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Load models and metadata
    models, results, feature_info = load_models_and_metadata()
    
    if models is None or not models:
        st.error("❌ No models found! Please run the training script first.")
        st.info("Make sure the 'saved_models' directory exists with trained models.")
        return
    
    # Header
    st.title("🚛 Truck Production Prediction System")
    st.markdown("### Interactive prediction tool for mining truck production optimization")
    
    # Sidebar for model selection and info
    with st.sidebar:
        st.header("🔧 Model Configuration")
        
        # Model selection
        model_names = list(models.keys())
        selected_model = st.selectbox(
            "Select Prediction Model:",
            model_names,
            index=0
        )
        
        if results:
            st.markdown("### 📊 Model Performance")
            model_metrics = results[selected_model]['metrics']
            
            st.markdown(f"""
            <div class="metric-card">
                <strong>R² Score:</strong> {model_metrics['R2']:.4f}<br>
                <strong>MAPE:</strong> {model_metrics['MAPE']:.2f}%<br>
                <strong>VAF:</strong> {model_metrics.get('VAF', 'N/A')}<br>
                <strong>NASH:</strong> {model_metrics.get('NASH', 'N/A')}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### ℹ️ About the Models")
        model_descriptions = {
            'Base_ELM': 'Standard Extreme Learning Machine',
            'ICA_ELM': 'ELM optimized with Imperialist Competitive Algorithm',
            'HBO_ELM': 'ELM optimized with Heap-Based Optimizer',
            'MFO_ELM': 'ELM optimized with Moth-Flame Optimization',
            'BOA_ELM': 'ELM optimized with Butterfly Optimization Algorithm'
        }
        
        if selected_model in model_descriptions:
            st.info(model_descriptions[selected_model])
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎛️ Input Parameters")
        
        # Create input form
        with st.form("prediction_form"):
            # Row 1
            row1_col1, row1_col2, row1_col3 = st.columns(3)
            
            with row1_col1:
                truck_model = st.selectbox(
                    "Truck Model:",
                    feature_info['truck_models'],
                    help="Select the truck model for production"
                )
            
            with row1_col2:
                nominal_tonnage = st.slider(
                    "Nominal Tonnage (tonnes):",
                    min_value=50.0,
                    max_value=200.0,
                    value=100.0,
                    step=5.0,
                    help="Truck carrying capacity"
                )
            
            with row1_col3:
                material_type = st.selectbox(
                    "Material Type:",
                    feature_info['material_types'],
                    help="Type of material being transported"
                )
            
            # Row 2
            row2_col1, row2_col2 = st.columns(2)
            
            with row2_col1:
                fixed_time = st.slider(
                    "Fixed Time (hours):",
                    min_value=1.0,
                    max_value=20.0,
                    value=8.0,
                    step=0.5,
                    help="Fixed operational time (FH+EH)"
                )
            
            with row2_col2:
                variable_time = st.slider(
                    "Variable Time (hours):",
                    min_value=0.5,
                    max_value=15.0,
                    value=5.0,
                    step=0.5,
                    help="Variable operational time (DT+Q+LT)"
                )
            
            # Row 3
            row3_col1, row3_col2 = st.columns(2)
            
            with row3_col1:
                number_of_loads = st.slider(
                    "Number of Loads:",
                    min_value=1,
                    max_value=100,
                    value=20,
                    step=1,
                    help="Total number of loads to transport"
                )
            
            with row3_col2:
                cycle_distance = st.slider(
                    "Cycle Distance (km):",
                    min_value=0.1,
                    max_value=20.0,
                    value=5.0,
                    step=0.1,
                    help="Round-trip distance for each cycle"
                )
            
            # Predict button
            predict_button = st.form_submit_button(
                "🚀 Predict Production",
                use_container_width=True
            )
    
    with col2:
        st.header("📈 Prediction Results")
        
        if predict_button:
            # Prepare input parameters
            input_params = {
                'truck_model': truck_model,
                'nominal_tonnage': nominal_tonnage,
                'material_type': material_type,
                'fixed_time': fixed_time,
                'variable_time': variable_time,
                'number_of_loads': number_of_loads,
                'cycle_distance': cycle_distance
            }
            
            # Make prediction
            try:
                selected_model_obj = models[selected_model]
                prediction = make_prediction(
                    selected_model_obj, 
                    **input_params, 
                    feature_info=feature_info
                )
                
                # Display prediction
                st.markdown(f"""
                <div class="prediction-card">
                    <h2>🎯 Predicted Production</h2>
                    <h1>{prediction:.2f} tonnes</h1>
                    <p>Using {selected_model}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional metrics
                efficiency = prediction / (fixed_time + variable_time)
                tons_per_load = prediction / number_of_loads
                
                st.metric("Production Efficiency", f"{efficiency:.2f} tonnes/hour")
                st.metric("Average per Load", f"{tons_per_load:.2f} tonnes/load")
                
                # Compare with all models
                st.subheader("📊 All Model Predictions")
                comparison_data = []
                
                for model_name, model in models.items():
                    pred = make_prediction(model, **input_params, feature_info=feature_info)
                    r2 = results[model_name]['metrics']['R2'] if results else 0
                    comparison_data.append({
                        'Model': model_name,
                        'Prediction (tonnes)': f"{pred:.2f}",
                        'R² Score': f"{r2:.4f}"
                    })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    
    # Additional analysis sections
    if results:
        st.header("📊 Model Performance Analysis")
        
        # Model comparison chart
        comparison_chart = create_model_comparison_chart(results)
        if comparison_chart:
            st.plotly_chart(comparison_chart, use_container_width=True)
    
    # Parameter sensitivity analysis
    if predict_button and models:
        st.header("🔍 Parameter Sensitivity Analysis")
        st.markdown("See how production changes when varying each parameter:")
        
        sensitivity_chart = create_sensitivity_chart(input_params, models, feature_info)
        st.plotly_chart(sensitivity_chart, use_container_width=True)
    
    # Additional information
    with st.expander("ℹ️ How to Use This App"):
        st.markdown("""
        ### 🎯 Quick Start Guide
        
        1. **Select a Model**: Choose from the trained models in the sidebar
        2. **Set Parameters**: Adjust the operational parameters using the sliders
        3. **Predict**: Click "Predict Production" to get results
        4. **Analyze**: Review the sensitivity analysis to understand parameter impacts
        
        ### 📊 Understanding the Results
        
        - **Predicted Production**: Main output in tonnes
        - **Efficiency**: Production per hour of operation
        - **Per Load Average**: Production efficiency per individual load
        - **Model Comparison**: See how different models perform on your inputs
        
        ### 🔧 Model Types
        
        - **Base_ELM**: Standard implementation
        - **ICA_ELM**: Optimized with Imperialist Competitive Algorithm
        - **HBO_ELM**: Optimized with Heap-Based Optimizer  
        - **MFO_ELM**: Optimized with Moth-Flame Optimization
        - **BOA_ELM**: Optimized with Butterfly Optimization Algorithm
        
        ### 📈 Tips for Optimization
        
        - Use the sensitivity analysis to identify critical parameters
        - Compare predictions across different models for validation
        - Experiment with different parameter combinations
        - Focus on parameters with high sensitivity indices
        """)

if __name__ == "__main__":
    main()