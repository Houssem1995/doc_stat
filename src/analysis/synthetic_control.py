from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

@dataclass
class SyntheticControlResult:
    """Class to store synthetic control analysis results."""
    outcome: str
    treatment_effect: float
    pre_rmse: float
    weights: np.ndarray
    treated_values: pd.Series
    synthetic_values: pd.Series
    time_values: pd.Series
    treatment_time: int
    metrics: Dict[str, float]

class SyntheticControlAnalyzer:
    """Class for performing synthetic control analysis."""
    
    @staticmethod
    def perform_analysis(
        data: pd.DataFrame,
        treatment_var: str,
        outcome_var: str,
        time_var: str,
        treatment_time: int,
        covariates: Optional[List[str]] = None
    ) -> SyntheticControlResult:
        """
        Perform synthetic control analysis.
        """
        # Validate inputs
        if data[treatment_var].nunique() != 2:
            raise ValueError(f"Treatment variable must be binary (0/1). Found {data[treatment_var].nunique()} unique values.")
        
        # Split data into treated and control groups
        treated_data = data[data[treatment_var] == 1].copy()
        control_data = data[data[treatment_var] == 0].copy()
        
        if len(treated_data) == 0:
            raise ValueError("No treated units found in the data.")
        if len(control_data) == 0:
            raise ValueError("No control units found in the data.")
        
        # Create time periods
        pre_period = data[data[time_var] < treatment_time]
        post_period = data[data[time_var] >= treatment_time]
        
        if len(pre_period) == 0:
            raise ValueError("No pre-treatment periods found.")
        if len(post_period) == 0:
            raise ValueError("No post-treatment periods found.")
        
        # Prepare data matrices
        treated_outcomes = treated_data.pivot(
            index=time_var,
            columns=None,
            values=outcome_var
        ).sort_index()
        
        control_outcomes = control_data.pivot(
            index=time_var,
            columns=treatment_var,
            values=outcome_var
        ).sort_index()
        
        # Get pre-treatment data
        pre_treated = treated_outcomes[treated_outcomes.index < treatment_time]
        pre_control = control_outcomes[control_outcomes.index < treatment_time]
        
        if len(pre_treated) == 0 or len(pre_control) == 0:
            raise ValueError("Insufficient pre-treatment data for both treated and control units.")
        
        # Calculate weights
        Z1 = pre_treated.values
        Z0 = pre_control.values
        
        if Z0.size == 0:
            raise ValueError("No control units available for comparison.")
        
        weights = SyntheticControlAnalyzer._calculate_weights(Z1, Z0)
        
        # Generate synthetic control
        synthetic_values = np.dot(control_outcomes.values, weights)
        synthetic_series = pd.Series(
            synthetic_values, 
            index=control_outcomes.index,
            name='synthetic'
        )
        
        # Combine results
        results = pd.DataFrame({
            'time': treated_outcomes.index,
            'treated': treated_outcomes.values.flatten(),
            'synthetic': synthetic_series
        })
        
        # Calculate treatment effect
        post_effect = (
            results[results['time'] >= treatment_time]['treated'].mean() -
            results[results['time'] >= treatment_time]['synthetic'].mean()
        )
        
        # Calculate pre-treatment RMSE
        pre_rmse = np.sqrt(np.mean(
            (results[results['time'] < treatment_time]['treated'] -
             results[results['time'] < treatment_time]['synthetic'])**2
        ))
        
        # Calculate R-squared
        r_squared = SyntheticControlAnalyzer._calculate_r_squared(
            results['treated'].values,
            results['synthetic'].values
        )
        
        return SyntheticControlResult(
            outcome=outcome_var,
            treatment_effect=post_effect,
            pre_rmse=pre_rmse,
            weights=weights,
            treated_values=results['treated'],
            synthetic_values=results['synthetic'],
            time_values=results['time'],
            treatment_time=treatment_time,
            metrics={
                'pre_period_rmse': pre_rmse,
                'treatment_effect': post_effect,
                'r_squared': r_squared,
                'n_control_units': len(control_data),
                'n_time_periods': len(results)
            }
        )
    
    @staticmethod
    def _calculate_weights(Z1: np.ndarray, Z0: np.ndarray) -> np.ndarray:
        """Calculate optimal weights using quadratic programming."""
        # Ensure matrices are properly shaped
        if Z0.ndim == 1:
            Z0 = Z0.reshape(-1, 1)
        if Z1.ndim == 1:
            Z1 = Z1.reshape(-1, 1)
        
        n_controls = Z0.shape[1]
        if n_controls == 0:
            raise ValueError("No control units available for comparison.")
        
        def objective(w):
            return np.sum((Z1 - np.dot(Z0, w))**2)
        
        # Constraints: weights sum to 1 and are non-negative
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        ]
        bounds = [(0, 1) for _ in range(n_controls)]
        
        # Initial weights (equal weighting)
        w_initial = np.ones(n_controls) / n_controls
        
        # Optimize
        result = minimize(
            objective,
            w_initial,
            method='SLSQP',
            constraints=constraints,
            bounds=bounds,
            options={'ftol': 1e-8}
        )
        
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")
        
        return result.x
    
    @staticmethod
    def _calculate_r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate R-squared for goodness of fit."""
        ss_res = np.sum((actual - predicted)**2)
        ss_tot = np.sum((actual - np.mean(actual))**2)
        return 1 - (ss_res / ss_tot)
    
    @staticmethod
    def create_plots(result: SyntheticControlResult):
        """Create plots for synthetic control analysis."""
        # Trend plot
        fig = go.Figure()
        
        # Add treated unit
        fig.add_trace(go.Scatter(
            x=result.time_values,
            y=result.treated_values,
            name='Treated Unit',
            line=dict(color='blue')
        ))
        
        # Add synthetic control
        fig.add_trace(go.Scatter(
            x=result.time_values,
            y=result.synthetic_values,
            name='Synthetic Control',
            line=dict(color='red', dash='dash')
        ))
        
        # Add vertical line for treatment time
        fig.add_vline(
            x=result.treatment_time,
            line_dash="dash",
            annotation_text="Treatment Time"
        )
        
        fig.update_layout(
            title='Treated vs Synthetic Control Unit',
            xaxis_title='Time',
            yaxis_title=result.outcome,
            showlegend=True
        )
        
        st.plotly_chart(fig)
        
        # Gap plot
        gap = result.treated_values - result.synthetic_values
        fig_gap = go.Figure()
        
        fig_gap.add_trace(go.Scatter(
            x=result.time_values,
            y=gap,
            name='Gap',
            line=dict(color='green')
        ))
        
        fig_gap.add_hline(
            y=0,
            line_dash="dash"
        )
        
        fig_gap.add_vline(
            x=result.treatment_time,
            line_dash="dash",
            annotation_text="Treatment Time"
        )
        
        fig_gap.update_layout(
            title='Gap between Treated and Synthetic Control',
            xaxis_title='Time',
            yaxis_title='Difference',
            showlegend=True
        )
        
        st.plotly_chart(fig_gap) 