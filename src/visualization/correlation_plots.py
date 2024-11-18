import streamlit as st
import plotly.graph_objects as go
import numpy as np
from typing import Any

def create_correlation_plots(correlation_result: Any) -> None:
    """
    Create and display correlation analysis plots.
    
    Args:
        correlation_result: Result from correlation analysis
    """
    # Create heatmap
    fig = create_correlation_heatmap(
        correlation_result.correlation_matrix,
        correlation_result.p_value_matrix
    )
    st.plotly_chart(fig, use_container_width=True)

def create_correlation_heatmap(corr_matrix: np.ndarray, 
                             p_values: np.ndarray) -> go.Figure:
    """Create correlation heatmap with significance indicators."""
    fig = go.Figure()
    
    # Add heatmap
    fig.add_trace(go.Heatmap(
        z=corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        zmin=-1,
        zmax=1,
        colorscale='RdBu',
        colorbar=dict(title='Correlation')
    ))
    
    # Add significance annotations
    annotations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            if i != j:
                annotations.append(dict(
                    x=corr_matrix.columns[i],
                    y=corr_matrix.columns[j],
                    text=f"{corr_matrix.iloc[i, j]:.2f}<br>p={p_values.iloc[i, j]:.3f}",
                    showarrow=False,
                    font=dict(
                        color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black'
                    )
                ))
    
    fig.update_layout(
        title='Correlation Matrix',
        annotations=annotations,
        height=600,
        width=800
    )
    
    return fig 