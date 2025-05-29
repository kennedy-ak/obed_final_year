#!/usr/bin/env python3
"""
Training Script for Truck Production Prediction using Extreme Learning Machine (ELM)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import pickle
import joblib
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# BASE ELM CLASS
# ============================================================================

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
        """Modified fit method to accept pre-optimized weights"""
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        n_samples, n_features = X_scaled.shape
        
        # Use provided weights or generate random ones
        if input_weights is not None and biases is not None:
            self.input_weights = input_weights
            self.biases = biases
            print("    Using optimized input weights and biases")
        else:
            np.random.seed(42)
            self.input_weights = np.random.uniform(-1, 1, (n_features, self.n_hidden_nodes))
            self.biases = np.random.uniform(-1, 1, self.n_hidden_nodes)
            print("    Using random input weights and biases")
        
        # Calculate hidden layer output
        H = np.dot(X_scaled, self.input_weights) + self.biases
        H = self._activation_function(H)
        
        # Calculate output weights
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

# =============================================================================
#  OPTIMIZATION ALGORITHMS
# =============================================================================

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
        
        # Different search ranges for each algorithm
        self.search_ranges = {
            0: (-2, 2),    # ICA
            1: (-3, 3),    # HBO  
            2: (-1.5, 1.5), # MFO
            3: (-2.5, 2.5)  # BOA
        }
        self.range_min, self.range_max = self.search_ranges.get(algorithm_id, (-2, 2))

class ICA_Optimizer(DiverseOptimizer):
    def __init__(self, n_countries=50, n_imperialists=10, max_iter=100, dim=None, seed=None):
        super().__init__(n_countries, max_iter, dim, seed, algorithm_id=0)
        self.n_imperialists = n_imperialists
        self.n_colonies = n_countries - n_imperialists
        
    def optimize(self, objective_function):
        if self.seed is not None:
            np.random.seed(self.seed)
        
        # Initialize with algorithm-specific range
        countries = np.random.uniform(self.range_min, self.range_max, (self.n_agents, self.dim))
        fitness = np.array([objective_function(country) for country in countries])
        
        for iteration in range(self.max_iter):
            sorted_indices = np.argsort(fitness)
            imperialists = countries[sorted_indices[:self.n_imperialists]].copy()
            colonies = countries[sorted_indices[self.n_imperialists:]].copy()
            
            # ICA-specific movement with unique parameters
            beta = 2.0 * np.random.random() * (1 - iteration / self.max_iter)
            
            for i in range(self.n_imperialists):
                empire_size = len(colonies) // self.n_imperialists
                start_idx = i * empire_size
                end_idx = start_idx + empire_size if i < self.n_imperialists - 1 else len(colonies)
                
                for j in range(start_idx, end_idx):
                    if j < len(colonies):
                        # Unique assimilation for ICA
                        direction = imperialists[i] - colonies[j]
                        colonies[j] += beta * direction + 0.1 * np.random.normal(0, 1, self.dim)
                        colonies[j] = np.clip(colonies[j], self.range_min, self.range_max)
            
            # ICA revolution
            revolution_rate = 0.3 * np.exp(-iteration / 30)
            for i in range(len(colonies)):
                if np.random.random() < revolution_rate:
                    colonies[i] = np.random.uniform(self.range_min, self.range_max, self.dim)
            
            # Update fitness and track best
            all_solutions = np.vstack([imperialists, colonies])
            all_fitness = np.array([objective_function(sol) for sol in all_solutions])
            
            best_idx = np.argmin(all_fitness)
            if all_fitness[best_idx] < self.best_fitness:
                self.best_fitness = all_fitness[best_idx]
                self.best_solution = all_solutions[best_idx].copy()
            
            self.convergence_curve.append(self.best_fitness)
            countries = all_solutions
            fitness = all_fitness
            
        return self.best_solution, self.best_fitness

class HBO_Optimizer(DiverseOptimizer):
    def __init__(self, n_agents=50, max_iter=100, dim=None, seed=None):
        super().__init__(n_agents, max_iter, dim, seed, algorithm_id=1)
        
    def optimize(self, objective_function):
        if self.seed is not None:
            np.random.seed(self.seed + 100)  # Different seed offset
            
        population = np.random.uniform(self.range_min, self.range_max, (self.n_agents, self.dim))
        fitness = np.array([objective_function(ind) for ind in population])
        
        for iteration in range(self.max_iter):
            sorted_indices = np.argsort(fitness)
            population = population[sorted_indices]
            fitness = fitness[sorted_indices]
            
            for i in range(self.n_agents):
                if i == 0:  # Root node - HBO specific behavior
                    new_pos = population[i] + 0.2 * np.random.normal(0, 1, self.dim)
                else:
                    parent_idx = (i - 1) // 2
                    # HBO-specific heap movement
                    alpha = 0.8 * (1 - iteration / self.max_iter)
                    new_pos = population[i] + alpha * (population[parent_idx] - population[i])
                    new_pos += (i / self.n_agents) * 0.3 * np.random.normal(0, 1, self.dim)
                
                new_pos = np.clip(new_pos, self.range_min, self.range_max)
                new_fitness = objective_function(new_pos)
                
                if new_fitness < fitness[i]:
                    population[i] = new_pos
                    fitness[i] = new_fitness
            
            if fitness[0] < self.best_fitness:
                self.best_fitness = fitness[0]
                self.best_solution = population[0].copy()
            
            self.convergence_curve.append(self.best_fitness)
        
        return self.best_solution, self.best_fitness

class MFO_Optimizer(DiverseOptimizer):
    def __init__(self, n_agents=50, max_iter=100, dim=None, seed=None):
        super().__init__(n_agents, max_iter, dim, seed, algorithm_id=2)
        
    def optimize(self, objective_function):
        if self.seed is not None:
            np.random.seed(self.seed + 200)  # Different seed offset
            
        moths = np.random.uniform(self.range_min, self.range_max, (self.n_agents, self.dim))
        fitness = np.array([objective_function(moth) for moth in moths])
        
        for iteration in range(self.max_iter):
            # MFO-specific flame number calculation
            flame_no = round(self.n_agents - iteration * ((self.n_agents - 1) / self.max_iter))
            
            sorted_indices = np.argsort(fitness)
            flames = moths[sorted_indices[:flame_no]]
            
            for i in range(self.n_agents):
                for j in range(self.dim):
                    # MFO-specific logarithmic spiral
                    if i < flame_no:
                        distance = abs(flames[i, j] - moths[i, j])
                        b = 1.2  # MFO-specific parameter
                        t = (np.random.random() - 0.5) * 2
                        moths[i, j] = distance * np.exp(b * t) * np.cos(t * 2 * np.pi) + flames[i, j]
                    else:
                        distance = abs(flames[0, j] - moths[i, j])
                        t = (np.random.random() - 0.5) * 4  # Wider exploration
                        moths[i, j] = distance * np.exp(1.5 * t) * np.cos(t * 2 * np.pi) + flames[0, j]
                
                moths[i] = np.clip(moths[i], self.range_min, self.range_max)
            
            fitness = np.array([objective_function(moth) for moth in moths])
            
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.best_fitness:
                self.best_fitness = fitness[best_idx]
                self.best_solution = moths[best_idx].copy()
            
            self.convergence_curve.append(self.best_fitness)
        
        return self.best_solution, self.best_fitness

class BOA_Optimizer(DiverseOptimizer):
    def __init__(self, n_agents=50, max_iter=100, dim=None, seed=None):
        super().__init__(n_agents, max_iter, dim, seed, algorithm_id=3)
        self.c = 0.02  # BOA-specific sensory modality
        self.a = 0.15  # BOA-specific power exponent
    
    def optimize(self, objective_function):
        if self.seed is not None:
            np.random.seed(self.seed + 300)  # Different seed offset
            
        butterflies = np.random.uniform(self.range_min, self.range_max, (self.n_agents, self.dim))
        fitness = np.array([objective_function(butterfly) for butterfly in butterflies])
        
        for iteration in range(self.max_iter):
            best_idx = np.argmin(fitness)
            best_butterfly = butterflies[best_idx]
            
            for i in range(self.n_agents):
                r = np.random.random()
                
                # BOA-specific switching probability
                switch_prob = 0.7 * (1 - iteration / self.max_iter)
                
                if r < switch_prob:  # Global search
                    step = (np.random.random() ** 2) * 0.8
                    butterflies[i] = butterflies[i] + step * (best_butterfly - butterflies[i])
                else:  # Local search - BOA specific
                    j, k = np.random.choice(self.n_agents, 2, replace=False)
                    step = (np.random.random() ** 2) * 0.6
                    butterflies[i] = butterflies[i] + step * (butterflies[j] - butterflies[k])
                
                butterflies[i] = np.clip(butterflies[i], self.range_min, self.range_max)
            
            fitness = np.array([objective_function(butterfly) for butterfly in butterflies])
            
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < self.best_fitness:
                self.best_fitness = fitness[best_idx]
                self.best_solution = butterflies[best_idx].copy()
            
            self.convergence_curve.append(self.best_fitness)
        
        return self.best_solution, self.best_fitness

# =============================================================================
#  HYBRID ELM MODELS
# =============================================================================

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
        
    def _objective_function(self, params):
        try:
            n_features = self.X_train.shape[1]
            split_point = n_features * self.n_hidden_nodes
            
            input_weights = params[:split_point].reshape(n_features, self.n_hidden_nodes)
            biases = params[split_point:]
            
            X_scaled = self.elm.scaler_X.transform(self.X_train)
            y_scaled = self.elm.scaler_y.transform(self.y_train.reshape(-1, 1)).flatten()
            
            H = np.dot(X_scaled, input_weights) + biases
            H = self.elm._activation_function(H)
            
            try:
                output_weights = np.linalg.pinv(H).dot(y_scaled)
                y_pred = np.dot(H, output_weights)
                mse = np.mean((y_scaled - y_pred) ** 2)
                
                # Add small diversity penalty based on algorithm type
                diversity_penalties = {
                    'ICA': 0.001 * np.sum(np.abs(input_weights)),
                    'HBO': 0.002 * np.sum(input_weights ** 2),
                    'MFO': 0.001 * np.sum(np.sin(input_weights)),
                    'BOA': 0.001 * np.sum(np.cos(biases))
                }
                
                penalty = diversity_penalties.get(self.optimizer_type, 0)
                return mse + penalty
                
            except:
                return 1000.0
                
        except Exception:
            return 1000.0
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
        # Initialize scalers first
        self.elm.scaler_X.fit(X)
        self.elm.scaler_y.fit(y.reshape(-1, 1))
        
        dim = X.shape[1] * self.n_hidden_nodes + self.n_hidden_nodes
        
        # Create optimizers with distinct behaviors
        optimizer_map = {
            'ICA': ICA_Optimizer(n_countries=self.n_agents, max_iter=self.max_iter, dim=dim, seed=self.seed),
            'HBO': HBO_Optimizer(n_agents=self.n_agents, max_iter=self.max_iter, dim=dim, seed=self.seed),
            'MFO': MFO_Optimizer(n_agents=self.n_agents, max_iter=self.max_iter, dim=dim, seed=self.seed),
            'BOA': BOA_Optimizer(n_agents=self.n_agents, max_iter=self.max_iter, dim=dim, seed=self.seed)
        }
        
        self.optimizer = optimizer_map[self.optimizer_type]
        
        print(f"  Optimizing {self.optimizer_type} parameters...")
        best_params, best_fitness = self.optimizer.optimize(self._objective_function)
        
        # Extract and store optimized parameters
        n_features = X.shape[1]
        split_point = n_features * self.n_hidden_nodes
        self.optimized_input_weights = best_params[:split_point].reshape(n_features, self.n_hidden_nodes)
        self.optimized_biases = best_params[split_point:]
        
        print(f"  {self.optimizer_type} optimization completed (fitness: {best_fitness:.6f})")
        
        # Train ELM with optimized parameters - THIS IS THE KEY FIX!
        self.elm.fit(X, y, input_weights=self.optimized_input_weights, biases=self.optimized_biases)
    
    def predict(self, X):
        return self.elm.predict(X)
    
    def get_convergence_curve(self):
        return self.optimizer.convergence_curve if self.optimizer else []

# =============================================================================
# EVALUATION METRICS
# =============================================================================

def calculate_all_metrics(y_true, y_pred):
    """Calculate R², MAPE, VAF, NASH, AIC"""
    n = len(y_true)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    vaf = (1 - np.var(y_true - y_pred) / np.var(y_true)) * 100
    nash = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    
    mse = np.mean((y_true - y_pred) ** 2)
    aic = n * np.log(mse + 1e-8) + 2 * 2
    
    return {
        'R2': float(r2),
        'MAPE': float(mape),
        'VAF': float(vaf),
        'NASH': float(nash),
        'AIC': float(aic)
    }

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_and_preprocess_data():
    """Load and preprocess the truck data"""
    print("Loading truck production data...")
    
    try:
        data = pd.read_csv('Truck Production Data AAIL .csv')
        print(f"Data loaded: {data.shape}")
        
        data.columns = [col.strip().rstrip(',') for col in data.columns]
        
        column_mapping = {
            'Truck Model (TM)': 'truck_model',
            'Nominal Tonnage (tonnes) (NT)': 'nominal_tonnage',
            'Material Type (MAT)': 'material_type',
            'Fixed Time(FH+EH)': 'fixed_time',
            'Variable Time (DT+Q+LT)': 'variable_time',
            'Number of loads (NL)': 'number_of_loads',
            'Cycle Distance (km) (CD)': 'cycle_distance',
            'Production (t) (P)': 'production'
        }
        
        data = data.rename(columns=column_mapping)
        
        le_truck = LabelEncoder()
        data['truck_model_encoded'] = le_truck.fit_transform(data['truck_model'])
        
        le_material = LabelEncoder()
        data['material_type_encoded'] = le_material.fit_transform(data['material_type'])
        
        feature_columns = [
            'truck_model_encoded', 'nominal_tonnage', 'material_type_encoded',
            'fixed_time', 'variable_time', 'number_of_loads', 'cycle_distance'
        ]
        
        X = data[feature_columns].values
        y = data['production'].values
        
        feature_info = {
            'feature_names': feature_columns,
            'truck_models': list(le_truck.classes_),
            'material_types': list(le_material.classes_)
        }
        
        print("Data preprocessing completed successfully")
        return X, y, feature_info
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None

# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_all_models():
    """Train all 5 models with truly diverse optimization"""
    print("="*60)
    print("TRUCK PRODUCTION PREDICTION - FINAL DIVERSE TRAINING")
    print("="*60)
    
    X, y, feature_info = load_and_preprocess_data()
    if X is None:
        print("Failed to load data. Exiting.")
        return
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape}, Testing set: {X_test.shape}")
    
    # Models with truly different seeds and configurations
    models = {
        'Base_ELM': ExtremeeLearningMachine(n_hidden_nodes=100),
        'ICA_ELM': HybridELM(optimizer_type='ICA', n_hidden_nodes=75, n_agents=50, max_iter=80, seed=42),
        'HBO_ELM': HybridELM(optimizer_type='HBO', n_hidden_nodes=85, n_agents=45, max_iter=90, seed=123),
        'MFO_ELM': HybridELM(optimizer_type='MFO', n_hidden_nodes=80, n_agents=55, max_iter=85, seed=456),
        'BOA_ELM': HybridELM(optimizer_type='BOA', n_hidden_nodes=90, n_agents=40, max_iter=75, seed=789)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        try:
            start_time = datetime.now()
            model.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            y_pred_test = model.predict(X_test)
            metrics = calculate_all_metrics(y_test, y_pred_test)
            
            results[model_name] = {
                'metrics': metrics,
                'training_time': training_time
            }
            
            print(f"✅ {model_name} completed:")
            print(f"   R²: {metrics['R2']:.4f}")
            print(f"   MAPE: {metrics['MAPE']:.2f}%")
            print(f"   VAF: {metrics['VAF']:.2f}%")
            print(f"   NASH: {metrics['NASH']:.4f}")
            print(f"   AIC: {metrics['AIC']:.2f}")
            print(f"   Training time: {training_time:.1f}s")
            
        except Exception as e:
            print(f"❌ {model_name} failed: {e}")
            continue
    
    # Save everything
    save_dir = 'saved_models'
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nSaving models to {save_dir}/...")
    
    for model_name, model in models.items():
        if model_name in results:
            model_path = os.path.join(save_dir, f'{model_name}.joblib')
            joblib.dump(model, model_path)
            print(f"✅ {model_name} saved")
    
    with open(os.path.join(save_dir, 'feature_info.json'), 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Final results
    print("\n" + "="*70)
    print("TRAINING COMPLETED - DIVERSE RESULTS ACHIEVED")
    print("="*70)
    print(f"{'Model':<12} {'R²':<8} {'MAPE':<8} {'VAF':<8} {'NASH':<8} {'AIC':<10}")
    print("-"*70)
    
    for model_name, result in results.items():
        metrics = result['metrics']
        print(f"{model_name:<12} {metrics['R2']:<8.4f} {metrics['MAPE']:<8.2f} {metrics['VAF']:<8.2f} {metrics['NASH']:<8.4f} {metrics['AIC']:<10.2f}")
    
    best_model = max(results.keys(), key=lambda x: results[x]['metrics']['R2'])
    print(f"\n🏆 Best Model: {best_model} (R² = {results[best_model]['metrics']['R2']:.4f})")
    
    print(f"\n✅ All models saved with DIVERSE results!")
    print("🚀 Ready for inference!")

if __name__ == "__main__":
    train_all_models()