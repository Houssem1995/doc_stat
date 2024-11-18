import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
import pandas as pd
from typing import Dict
from src.analysis.normality import NormalityTestResult

def create_multiple_normality_plots(data: pd.DataFrame, 
                                  results: Dict[str, NormalityTestResult]) -> None:
    """Create normality plots for multiple variables."""
    for variable, result in results.items():
        st.write(f"## Analysis for: {variable}")
        
        # Display test results
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sample Size", result.sample_size)
        with col2:
            st.metric("Missing Values", result.missing_values)
        with col3:
            st.metric("Normality", "✅ Normal" if result.is_normal else "❌ Non-normal")
        
        # Create plots
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Q-Q Plot', 'Distribution Plot'))
        
        # Add Q-Q plot
        qq_traces = create_qq_plot_traces(data[variable].dropna())
        for trace in qq_traces:
            fig.add_trace(trace, row=1, col=1)
        
        # Add histogram and density plot
        hist_traces = create_histogram_traces(data[variable].dropna())
        for trace in hist_traces:
            fig.add_trace(trace, row=1, col=2)
        
        fig.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display recommendations
        st.info(result.recommendation)
        st.markdown("---")

def create_qq_plot_traces(data: np.ndarray) -> list:
    """Create Q-Q plot traces."""
    qq = stats.probplot(data)
    
    traces = []
    # Data points
    traces.append(go.Scatter(
        x=qq[0][0],
        y=qq[0][1],
        mode='markers',
        name='Data',
        marker=dict(color='blue')
    ))
    
    # Reference line
    line_x = np.array([min(qq[0][0]), max(qq[0][0])])
    line_y = qq[1][1] + qq[1][0] * line_x
    traces.append(go.Scatter(
        x=line_x,
        y=line_y,
        mode='lines',
        name='Normal Line',
        line=dict(color='red', dash='dash')
    ))
    
    return traces

def create_histogram_traces(data: np.ndarray) -> list:
    """Create histogram and density plot traces."""
    traces = []
    
    # Histogram
    traces.append(go.Histogram(
        x=data,
        name='Histogram',
        nbinsx=30,
        histnorm='probability density'
    ))
    
    # Density plot
    kde_x, kde_y = calculate_kde(data)
    traces.append(go.Scatter(
        x=kde_x,
        y=kde_y,
        name='Density',
        line=dict(color='red')
    ))
    
    return traces

def calculate_kde(data: np.ndarray) -> tuple:
    """Calculate Kernel Density Estimation."""
    kernel = stats.gaussian_kde(data)
    x_range = np.linspace(min(data), max(data), 100)
    return x_range, kernel(x_range) 