import unittest
import pandas as pd
import numpy as np
from src.visualization.normality_plots import create_normality_plots
from src.visualization.correlation_plots import create_correlation_plots
from src.visualization.statistical_plots import create_statistical_plots
from unittest.mock import patch

class TestVisualization(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        np.random.seed(42)
        self.normal_data = np.random.normal(0, 1, 1000)
        self.df = pd.DataFrame({
            'var1': np.random.normal(0, 1, 100),
            'var2': np.random.normal(0, 1, 100),
            'group': ['A'] * 50 + ['B'] * 50
        })

    @patch('streamlit.plotly_chart')
    def test_normality_plots(self, mock_plotly_chart):
        """Test normality plots creation."""
        create_normality_plots(self.normal_data)
        self.assertEqual(mock_plotly_chart.call_count, 2)  # QQ plot and histogram

    @patch('streamlit.plotly_chart')
    def test_correlation_plots(self, mock_plotly_chart):
        """Test correlation plots creation."""
        corr_result = Mock(
            correlation_matrix=self.df[['var1', 'var2']].corr(),
            p_value_matrix=pd.DataFrame([[1, 0.5], [0.5, 1]]),
            method='pearson'
        )
        create_correlation_plots(corr_result)
        mock_plotly_chart.assert_called()

    @patch('streamlit.plotly_chart')
    def test_statistical_plots(self, mock_plotly_chart):
        """Test statistical plots creation."""
        test_result = Mock(
            test_name="Independent t-test",
            groups={'A': self.df[self.df['group'] == 'A']['var1'],
                   'B': self.df[self.df['group'] == 'B']['var1']}
        )
        create_statistical_plots(test_result, "Two Groups (T-test/Mann-Whitney)")
        mock_plotly_chart.assert_called()

if __name__ == '__main__':
    unittest.main() 