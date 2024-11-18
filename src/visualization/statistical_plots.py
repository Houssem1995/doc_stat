import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import Any

def create_statistical_plots(test_result: Any, test_type: str) -> None:
    """
    Create and display statistical test plots.
    
    Args:
        test_result: Result from statistical test
        test_type: Type of statistical test performed
    """
    if "Two Groups" in test_type or "Multiple Groups" in test_type:
        fig = create_box_plot(test_result)
    else:  # Categorical test
        fig = create_categorical_plot(test_result)
    
    st.plotly_chart(fig, use_container_width=True)

def create_box_plot(test_result: Any) -> go.Figure:
    """Create box plot for group comparisons."""
    # Prepare data for plotting
    data = []
    for group_name, group_data in test_result.groups.items():
        data.append(go.Box(
            y=group_data,
            name=group_name,
            boxpoints='outliers'
        ))
    
    fig = go.Figure(data=data)
    
    fig.update_layout(
        title=f'{test_result.test_name} - Group Comparisons',
        yaxis_title='Value',
        xaxis_title='Groups',
        showlegend=False,
        height=500
    )
    
    return fig

def create_categorical_plot(test_result: Any) -> go.Figure:
    """Create heatmap for categorical data."""
    contingency = test_result.groups['contingency']
    
    fig = px.imshow(
        contingency,
        labels=dict(color="Count"),
        color_continuous_scale="Viridis",
        aspect="auto"
    )
    
    fig.update_layout(
        title='Contingency Table Heatmap',
        height=500
    )
    
    return fig 