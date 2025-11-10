# src/agents/tesla_oscillation_agent.py
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
from scipy.optimize import curve_fit
from agents.agent_base import BaseAgent

def sum_damped_cos(t, *params):
    """
    params are groups of (A_m, lambda_m, omega_m, phi_m) repeated M times
    returns sum_m A_m * exp(-lambda_m * t) * cos(omega_m * t + phi_m)
    """
    t = np.asarray(t)
    M = int(len(params) / 4)
    y = np.zeros_like(t, dtype=float)
    for m in range(M):
        A = params[4*m + 0]
        lam = params[4*m + 1]
        omega = params[4*m + 2]
        phi = params[4*m + 3]
        y += A * np.exp(-lam * t) * np.cos(omega * t + phi)
    return y

class TeslaOscillationAgent(BaseAgent):
    """
    Fit sum of damped oscillators:
      y_hat(t) = sum_m A_m e^{-lambda_m t} cos(omega_m t + phi_m)
    Fit via nonlinear least squares (scipy.curve_fit).
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.M = int(self.config.get("M", 2))
        self.maxfev = int(self.config.get("maxfev", 50000))  # Increased max function evaluations
        self.params = None
        self.fit_success = False
        self.fit_error = None

    def fit(self, df: pd.DataFrame, target_col: str = "mean_temp"):
        try:
            y = df[target_col].astype(float).values
            t = np.arange(len(y))
            M = self.M
            
            # Normalize the data to help with fitting
            y_mean = np.mean(y)
            y_std = np.std(y) or 1.0
            y_norm = (y - y_mean) / y_std
            
            # Initial parameter guesses
            p0 = []
            for m in range(M):
                # A, lambda, omega, phi
                p0 += [
                    0.5,                        # amplitude
                    0.01,                       # damping
                    2 * np.pi * (m + 1) / 12.0, # frequency (seasonal)
                    0.0                         # phase
                ]
                
            # Set bounds to keep parameters reasonable
            bounds = ([-np.inf] * len(p0), [np.inf] * len(p0))
            
            # Run curve fitting with increased maxfev and better error handling
            popt, _ = curve_fit(
                sum_damped_cos, 
                t, 
                y_norm, 
                p0=p0, 
                maxfev=self.maxfev,
                bounds=bounds,
                method='trf'  # Trust Region Reflective algorithm
            )
            
            # Store parameters and denormalize amplitude
            self.params = popt
            self.params[0::4] = [a * y_std for a in popt[0::4]]  # Scale amplitudes back
            self.fit_success = True
            self.fit_error = None
            
        except Exception as e:
            self.fit_success = False
            self.fit_error = str(e)
            raise RuntimeError(f"Curve fit failed after {self.maxfev} iterations: {str(e)}")
            
        return self
        return self

    def transform(self, df: pd.DataFrame):
        if not self.fit_success:
            raise RuntimeError("Agent not successfully fit yet")
            
        t = np.arange(len(df))
        yhat = sum_damped_cos(t, *self.params)
        
        # Create output with predictions and components
        out = df.copy().reset_index(drop=True)
        out["tesla_osc_pred"] = yhat
        
        # Add individual components
        for m in range(self.M):
            A = self.params[4*m + 0]
            lam = self.params[4*m + 1]
            omega = self.params[4*m + 2]
            phi = self.params[4*m + 3]
            component = A * np.exp(-lam * t) * np.cos(omega * t + phi)
            out[f"osc_component_{m+1}"] = component
            
        return out
        
    def plot(self):
        """Generate a plot of the fitted model and its components"""
        if not self.fit_success:
            return None
            
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create time points for smooth plotting
        t = np.linspace(0, 24, 1000)  # Assume 24 time points for smooth plotting
        
        # Create subplots
        fig = make_subplots(rows=self.M + 1, cols=1, 
                           subplot_titles=[f"Oscillator {i+1}" for i in range(self.M + 1)])
        
        # Plot each component
        for m in range(self.M):
            A = self.params[4*m + 0]
            lam = self.params[4*m + 1]
            omega = self.params[4*m + 2]
            phi = self.params[4*m + 3]
            
            # Damped cosine component
            y_component = A * np.exp(-lam * t) * np.cos(omega * t + phi)
            
            # Envelope
            envelope = A * np.exp(-lam * t)
            
            # Add to plot
            fig.add_trace(go.Scatter(
                x=t, y=y_component,
                mode='lines',
                name=f'Oscillator {m+1} (ω={omega:.2f})',
                line=dict(width=2)
            ), row=m+1, col=1)
            
            # Add envelope
            fig.add_trace(go.Scatter(
                x=np.concatenate([t, t[::-1]]),
                y=np.concatenate([envelope, -envelope[::-1]]),
                fill='toself',
                fillcolor='rgba(0,100,80,0.1)',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ), row=m+1, col=1)
        
        # Plot the sum of all components
        y_total = np.zeros_like(t)
        for m in range(self.M):
            A = self.params[4*m + 0]
            lam = self.params[4*m + 1]
            omega = self.params[4*m + 2]
            phi = self.params[4*m + 3]
            y_total += A * np.exp(-lam * t) * np.cos(omega * t + phi)
            
        fig.add_trace(go.Scatter(
            x=t, y=y_total,
            mode='lines',
            name='Sum of Oscillators',
            line=dict(color='red', width=2)
        ), row=self.M+1, col=1)
        
        # Update layout
        fig.update_layout(
            height=300 * (self.M + 1),
            showlegend=True,
            title_text="Tesla Oscillation Analysis"
        )
        
        for i in range(1, self.M + 2):
            fig.update_xaxes(title_text="Time", row=i, col=1)
            fig.update_yaxes(title_text="Amplitude", row=i, col=1)
        
        return fig
