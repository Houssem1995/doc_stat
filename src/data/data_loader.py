import pandas as pd
import streamlit as st
from typing import Tuple, Optional
from config.settings import ALLOWED_EXTENSIONS

class DataLoader:
    @staticmethod
    def load_data(file) -> Optional[pd.DataFrame]:
        """Load data from uploaded file."""
        try:
            file_extension = f".{file.name.split('.')[-1].lower()}"
            
            if file_extension not in ALLOWED_EXTENSIONS:
                st.error(f"Unsupported file type. Please upload one of: {', '.join(ALLOWED_EXTENSIONS)}")
                return None
                
            if file_extension == '.csv':
                df = pd.read_csv(file)
            else:  # Excel files
                df = pd.read_excel(file)
                
            return df
            
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    
    @staticmethod
    def get_column_types(df: pd.DataFrame) -> Tuple[list, list]:
        """Identify numeric and categorical columns."""
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        return numeric_cols, categorical_cols
    
    @staticmethod
    def validate_data(df: pd.DataFrame) -> bool:
        """Validate loaded data."""
        if df.empty:
            st.error("The uploaded file is empty.")
            return False
            
        if df.columns.duplicated().any():
            st.error("Duplicate column names found. Please check your data.")
            return False
            
        return True 