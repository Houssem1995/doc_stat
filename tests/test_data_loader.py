import unittest
import pandas as pd
import numpy as np
from src.data.data_loader import DataLoader
from unittest.mock import Mock, patch
import io

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        """Set up test cases."""
        # Create sample data
        self.sample_data = pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5],
            'numeric2': [1.1, 2.2, 3.3, 4.4, 5.5],
            'categorical1': ['A', 'B', 'A', 'B', 'C'],
            'categorical2': ['X', 'Y', 'X', 'Y', 'Z']
        })

    def test_get_column_types(self):
        """Test column type identification."""
        numeric_cols, categorical_cols = DataLoader.get_column_types(self.sample_data)
        
        self.assertEqual(set(numeric_cols), {'numeric1', 'numeric2'})
        self.assertEqual(set(categorical_cols), {'categorical1', 'categorical2'})

    def test_validate_data_empty(self):
        """Test validation of empty dataframe."""
        empty_df = pd.DataFrame()
        self.assertFalse(DataLoader.validate_data(empty_df))

    def test_validate_data_duplicate_columns(self):
        """Test validation of dataframe with duplicate columns."""
        df_duplicate = pd.DataFrame({
            'col1': [1, 2],
            'col1': [3, 4]
        })
        self.assertFalse(DataLoader.validate_data(df_duplicate))

    @patch('streamlit.error')
    def test_load_data_invalid_extension(self, mock_st_error):
        """Test loading file with invalid extension."""
        mock_file = Mock()
        mock_file.name = 'test.txt'
        
        result = DataLoader.load_data(mock_file)
        self.assertIsNone(result)
        mock_st_error.assert_called()

if __name__ == '__main__':
    unittest.main() 