from dataclasses import dataclass
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

@dataclass
class RegressionResult:
    model_type: str
    r_squared: float
    adjusted_r_squared: float
    coefficients: Dict[str, float]
    p_values: Dict[str, float]
    std_errors: Dict[str, float]
    predictions: np.ndarray
    residuals: np.ndarray
    mse: float
    rmse: float
    f_statistic: Optional[float] = None
    f_pvalue: Optional[float] = None

class RegressionAnalyzer:
    @staticmethod
    def perform_regression(data: pd.DataFrame,
                         dependent_var: str,
                         independent_vars: List[str],
                         test_size: float = 0.2,
                         standardize: bool = True) -> RegressionResult:
        """Perform regression analysis."""
        # Prepare data
        X = data[independent_vars]
        y = data[dependent_var]
        
        if standardize:
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        
        # Add constant for statsmodels
        X = sm.add_constant(X)
        
        # Fit model
        model = sm.OLS(y, X).fit()
        
        # Calculate metrics
        predictions = model.predict(X)
        residuals = y - predictions
        mse = np.mean(residuals ** 2)
        rmse = np.sqrt(mse)
        
        # Prepare results
        return RegressionResult(
            model_type="Multiple" if len(independent_vars) > 1 else "Simple",
            r_squared=model.rsquared,
            adjusted_r_squared=model.rsquared_adj,
            coefficients=dict(zip(X.columns, model.params)),
            p_values=dict(zip(X.columns, model.pvalues)),
            std_errors=dict(zip(X.columns, model.bse)),
            predictions=predictions,
            residuals=residuals,
            mse=mse,
            rmse=rmse,
            f_statistic=model.fvalue,
            f_pvalue=model.f_pvalue
        ) 