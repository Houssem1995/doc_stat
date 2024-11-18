"""
DocStat - Statistical Analysis Tool
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A comprehensive statistical analysis tool built with Streamlit.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"

from src.data import DataLoader
from src.analysis import NormalityAnalyzer, CorrelationAnalyzer, StatisticalAnalyzer
from src.visualization import (
    create_multiple_normality_plots,
    create_correlation_plots,
    create_statistical_plots,
)

__all__ = [
    "DataLoader",
    "NormalityAnalyzer",
    "CorrelationAnalyzer",
    "StatisticalAnalyzer",
    "create_multiple_normality_plots",
    "create_correlation_plots",
    "create_statistical_plots",
]
