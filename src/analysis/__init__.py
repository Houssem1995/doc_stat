"""
Statistical analysis module containing various statistical tests and analyses.
"""

from .normality import NormalityAnalyzer
from .correlation import CorrelationAnalyzer
from .statistical import StatisticalAnalyzer

__all__ = [
    'NormalityAnalyzer',
    'CorrelationAnalyzer',
    'StatisticalAnalyzer'
] 