import unittest
import pandas as pd
import numpy as np
from src.analysis.normality import NormalityAnalyzer
from src.analysis.correlation import CorrelationAnalyzer
from src.analysis.statistical import StatisticalAnalyzer

class TestNormalityAnalyzer(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        # Create normal distribution data
        np.random.seed(42)
        self.normal_data = np.random.normal(0, 1, 1000)
        
        # Create non-normal distribution data
        self.non_normal_data = np.random.exponential(1, 1000)

    def test_shapiro_test_normal(self):
        """Test Shapiro-Wilk test with normal data."""
        result = NormalityAnalyzer.perform_shapiro_test(self.normal_data)
        self.assertTrue(result.is_normal)

    def test_shapiro_test_non_normal(self):
        """Test Shapiro-Wilk test with non-normal data."""
        result = NormalityAnalyzer.perform_shapiro_test(self.non_normal_data)
        self.assertFalse(result.is_normal)

class TestCorrelationAnalyzer(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        np.random.seed(42)
        self.df = pd.DataFrame({
            'var1': np.random.normal(0, 1, 100),
            'var2': np.random.normal(0, 1, 100),
            'var3': np.random.normal(0, 1, 100)
        })
        # Create correlation between var1 and var2
        self.df['var2'] = self.df['var1'] * 0.8 + np.random.normal(0, 0.2, 100)

    def test_correlation_calculation(self):
        """Test correlation calculation."""
        result = CorrelationAnalyzer.calculate_correlation(
            self.df, ['var1', 'var2', 'var3'], 'pearson'
        )
        self.assertGreater(abs(result.correlation_matrix.loc['var1', 'var2']), 0.7)
        self.assertLess(abs(result.correlation_matrix.loc['var1', 'var3']), 0.3)

class TestStatisticalAnalyzer(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        np.random.seed(42)
        self.df = pd.DataFrame({
            'target': np.random.normal(0, 1, 100),
            'group_two': ['A'] * 50 + ['B'] * 50,
            'group_multi': ['X'] * 30 + ['Y'] * 30 + ['Z'] * 40,
            'categorical': np.random.choice(['P', 'Q'], 100)
        })
        # Create group differences
        self.df.loc[self.df['group_two'] == 'B', 'target'] += 1

    def test_two_group_test(self):
        """Test two-group comparison."""
        result = StatisticalAnalyzer.perform_two_group_test(
            self.df, 'target', 'group_two'
        )
        self.assertTrue(result.significant)

    def test_multi_group_test(self):
        """Test multi-group comparison."""
        result = StatisticalAnalyzer.perform_multi_group_test(
            self.df, 'target', 'group_multi'
        )
        self.assertFalse(result.significant)

    def test_chi_square_test(self):
        """Test chi-square test."""
        result = StatisticalAnalyzer.perform_chi_square_test(
            self.df, 'categorical', 'group_two'
        )
        self.assertFalse(result.significant)

if __name__ == '__main__':
    unittest.main() 