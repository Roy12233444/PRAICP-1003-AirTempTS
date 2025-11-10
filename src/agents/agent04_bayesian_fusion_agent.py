# src/agents/bayesian_fusion_agent.py
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple
from agents.agent_base import BaseAgent
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

class BayesianFusionAgent(BaseAgent):
    """
    Fuse temperature data using Bayesian methods.
    
    If only one temperature column is provided, it will create a second synthetic model
    by adding noise to the original data to demonstrate the fusion process.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {})
        # Use the temperature column from config or detect it
        self.temp_col = self.config.get('temp_col', 'mean_temp')
        self._is_fitted = True  # This agent is stateless

    def fit(self, X, y=None, **kwargs):
        """
        Fit the fusion agent.
        
        Args:
            X: DataFrame containing the input data with temperature column
            y: Ignored, present for compatibility
            
        Returns:
            self: The fitted agent
        """
        # Store the temperature column name
        if self.temp_col not in X.columns:
            # Try to find a temperature column
            possible_cols = ['mean_temp', 'temperature', 'temp', 'Temperature', 'Mean_Temp']
            for col in possible_cols:
                if col in X.columns:
                    self.temp_col = col
                    break
            else:
                # If no temperature column found, use the first numeric column
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    self.temp_col = numeric_cols[0]
                else:
                    raise ValueError("No suitable temperature column found in the data")
        
        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """
        Apply Bayesian fusion to the input data.
        
        Args:
            X: DataFrame containing the input data with temperature values
            y: Ignored, present for compatibility
            
        Returns:
            DataFrame: Original data with added fused mean and variance columns
        """
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            self.fit(X)  # Fit if not already fitted
        
        # Make a copy of the input data
        result = X.copy()
        
        # Define possible column name patterns to check
        possible_columns = {
            'arima_mean': ['mu_arima', 'arima_mean', 'model1_temp'],
            'arima_var': ['var_arima', 'arima_var', 'model1_var'],
            'ml_mean': ['mu_ml', 'ml_mean', 'model2_temp'],
            'ml_var': ['var_ml', 'ml_var', 'model2_var']
        }
        
        # Check which columns exist in the dataframe
        found_columns = {}
        for col_type, possible_names in possible_columns.items():
            for name in possible_names:
                if name in X.columns:
                    found_columns[col_type] = name
                    break
        
        # If we have all required columns, use them
        if len(found_columns) == 4:
            muA_col = found_columns['arima_mean']
            varA_col = found_columns['arima_var']
            muM_col = found_columns['ml_mean']
            varM_col = found_columns['ml_var']
        else:
            # Create synthetic models if we don't have the required columns
            if self.temp_col not in X.columns:
                raise ValueError(f"Temperature column '{self.temp_col}' not found in the input data")
                
            # Add some noise to create a second model
            np.random.seed(42)  # For reproducibility
            noise = np.random.normal(0, 0.5, len(X))
            
            # Create synthetic models
            result['model1_temp'] = X[self.temp_col]
            result['model2_temp'] = X[self.temp_col] + noise
            
            # Add some variance estimates
            result['model1_var'] = 0.1 + 0.05 * np.abs(np.random.normal(0, 1, len(X)))
            result['model2_var'] = 0.15 + 0.1 * np.abs(np.random.normal(0, 1, len(X)))
            
            # Use these as our two models
            muA_col, varA_col = 'model1_temp', 'model1_var'
            muM_col, varM_col = 'model2_temp', 'model2_var'
        
        # Check if we have the required columns
        required_cols = [muA_col, varA_col, muM_col, varM_col]
        missing_cols = [col for col in required_cols if col not in result.columns]
        
        if missing_cols:
            # If we're missing columns but have the temperature column, create synthetic models
            if self.temp_col in result.columns and all('model' not in col for col in missing_cols):
                # Add some noise to create a second model
                np.random.seed(42)  # For reproducibility
                noise = np.random.normal(0, 0.5, len(result))
                
                # Create synthetic models
                result['model1_temp'] = result[self.temp_col]
                result['model2_temp'] = result[self.temp_col] + noise
                
                # Add some variance estimates
                result['model1_var'] = 0.1 + 0.05 * np.abs(np.random.normal(0, 1, len(result)))
                result['model2_var'] = 0.15 + 0.1 * np.abs(np.random.normal(0, 1, len(result)))
                
                # Update column references
                muA_col, varA_col = 'model1_temp', 'model1_var'
                muM_col, varM_col = 'model2_temp', 'model2_var'
                
                # Update required columns
                required_cols = [muA_col, varA_col, muM_col, varM_col]
                missing_cols = [col for col in required_cols if col not in result.columns]
                
                if missing_cols:
                    raise ValueError(f"Failed to create synthetic models. Missing columns: {missing_cols}")
            else:
                raise ValueError(f"Missing required columns for fusion: {missing_cols}. "
                              f"Available columns: {list(result.columns)}")
        
        # Perform the fusion
        fused_mean, fused_var = self.fuse(
            result[muA_col].values,
            result[varA_col].values,
            result[muM_col].values,
            result[varM_col].values
        )
        
        # Add the fused results to the output
        result['fused_mean'] = fused_mean
        result['fused_variance'] = fused_var
        result['fused_std'] = np.sqrt(fused_var)
        
        return result

    def fuse(self, muA, varA, muM, varM):
        """
        Fuse two Gaussian distributions using precision-weighted averaging.
        
        Args:
            muA: Mean of model A (can be array-like)
            varA: Variance of model A (can be array-like)
            muM: Mean of model M (can be array-like)
            varM: Variance of model M (can be array-like)
            
        Returns:
            Tuple of (fused_mean, fused_variance)
        """
        # Convert inputs to numpy arrays
        muA = np.asarray(muA, dtype=float)
        varA = np.asarray(varA, dtype=float)
        muM = np.asarray(muM, dtype=float)
        varM = np.asarray(varM, dtype=float)
        
        # Handle scalar inputs
        if np.isscalar(muA):
            muA = np.array([muA])
        if np.isscalar(varA):
            varA = np.array([varA])
        if np.isscalar(muM):
            muM = np.array([muM])
        if np.isscalar(varM):
            varM = np.array([varM])
        
        # Ensure all arrays have the same length
        if not (len(muA) == len(varA) == len(muM) == len(varM)):
            raise ValueError("All input arrays must have the same length")
        
        # Avoid division by zero or negative variances
        min_var = 1e-10  # Small positive value to avoid division by zero
        varA = np.maximum(varA, min_var)
        varM = np.maximum(varM, min_var)
        
        # Calculate precision (inverse variance)
        precA = 1.0 / varA
        precM = 1.0 / varM
        
        # Fuse using precision-weighted average
        fused_prec = precA + precM
        fused_var = 1.0 / fused_prec
        fused_mean = fused_var * (muA * precA + muM * precM)
        
        return fused_mean, fused_var

    def plot_results(self, df, figsize=(12, 6)):
        """
        Plot the original and fused temperature data.
        
        Args:
            df: DataFrame containing the data to plot
            figsize: Figure size (width, height)
            
        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot original temperature data if available
        if hasattr(self, 'temp_col') and self.temp_col in df.columns:
            ax.plot(df.index, df[self.temp_col], label='Original Temperature', 
                   color='blue', alpha=0.5)
        
        # Plot model predictions if available
        if 'model1_temp' in df.columns:
            ax.plot(df.index, df['model1_temp'], label='Model 1', 
                   color='green', linestyle='--', alpha=0.7)
        
        if 'model2_temp' in df.columns:
            ax.plot(df.index, df['model2_temp'], label='Model 2', 
                   color='red', linestyle='--', alpha=0.7)
        
        # Plot fused result if available
        if 'fused_mean' in df.columns:
            ax.plot(df.index, df['fused_mean'], label='Fused Result', 
                   color='black', linewidth=2)
            
            # Add uncertainty bands (mean ± 2*std)
            if 'fused_std' in df.columns:
                ax.fill_between(
                    df.index,
                    df['fused_mean'] - 2 * df['fused_std'],
                    df['fused_mean'] + 2 * df['fused_std'],
                    color='gray', alpha=0.2, label='95% Confidence'
                )
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Temperature')
        ax.set_title('Bayesian Fusion of Temperature Data')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        return fig
