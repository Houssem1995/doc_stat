from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import streamlit as st
from typing import List, Dict, Any, Optional

@dataclass
class SensitivityResult:
    """Class to store sensitivity analysis results."""
    analysis_type: str
    results_df: pd.DataFrame
    metrics: Dict[str, Any]
    additional_info: Optional[Dict[str, Any]] = None

class SensitivityAnalyzer:
    """Class for performing sensitivity analyses."""
    
    @staticmethod
    def analyze_missing_data_impact(data: pd.DataFrame,
                                  variables: List[str],
                                  percentages: tuple,
                                  methods: List[str],
                                  iterations: int) -> SensitivityResult:
        """Analyze the impact of missing data and imputation methods."""
        results = []
        original_data = data[variables].copy()
        
        for pct in range(percentages[0], percentages[1] + 1, 5):
            for method in methods:
                temp_results = []
                for _ in range(iterations):
                    # Create missing data
                    missing_data = original_data.copy()
                    for col in variables:
                        mask = np.random.choice(
                            [True, False],
                            size=len(data),
                            p=[pct/100, 1-pct/100]
                        )
                        missing_data.loc[mask, col] = np.nan
                    
                    # Impute missing values
                    imputed_data = SensitivityAnalyzer._impute_missing_values(missing_data, method)
                    
                    # Calculate metrics
                    metrics = {
                        'mean_diff': np.mean(np.abs(original_data - imputed_data)),
                        'std_diff': np.abs(original_data.std() - imputed_data.std()),
                        'corr': np.corrcoef(original_data.values.flatten(), imputed_data.values.flatten())[0,1]
                    }
                    temp_results.append(metrics)
                
                # Aggregate results
                avg_metrics = pd.DataFrame(temp_results).mean()
                results.append({
                    'percentage': pct,
                    'method': method,
                    **avg_metrics
                })
        
        results_df = pd.DataFrame(results)
        return SensitivityResult(
            analysis_type="Missing Data Impact",
            results_df=results_df,
            metrics={'iterations': iterations}
        )

    @staticmethod
    def analyze_outlier_impact(data: pd.DataFrame,
                             variables: List[str],
                             outlier_methods: List[str],
                             treatment_methods: List[str]) -> SensitivityResult:
        """Analyze the impact of different outlier detection and treatment methods."""
        results = []
        original_data = data[variables].copy()
        
        for method in outlier_methods:
            for treatment in treatment_methods:
                treated_data = original_data.copy()
                
                for var in variables:
                    # Detect outliers
                    if method == "Z-Score":
                        z_scores = np.abs(stats.zscore(treated_data[var]))
                        outliers = z_scores > 3
                    elif method == "IQR":
                        Q1 = treated_data[var].quantile(0.25)
                        Q3 = treated_data[var].quantile(0.75)
                        IQR = Q3 - Q1
                        outliers = (treated_data[var] < (Q1 - 1.5 * IQR)) | (treated_data[var] > (Q3 + 1.5 * IQR))
                    else:
                        continue
                    
                    # Treat outliers
                    if treatment == "Remove":
                        treated_data = treated_data[~outliers]
                    elif treatment == "Winsorize":
                        lower = np.percentile(treated_data[var], 5)
                        upper = np.percentile(treated_data[var], 95)
                        treated_data.loc[outliers, var] = np.clip(treated_data.loc[outliers, var], lower, upper)
                    elif treatment == "Cap":
                        treated_data.loc[outliers, var] = treated_data[var].mean()
                    
                    # Calculate impact metrics
                    metrics = {
                        'method': method,
                        'treatment': treatment,
                        'variable': var,
                        'original_mean': original_data[var].mean(),
                        'treated_mean': treated_data[var].mean(),
                        'original_std': original_data[var].std(),
                        'treated_std': treated_data[var].std(),
                        'outliers_pct': (outliers.sum() / len(data)) * 100
                    }
                    results.append(metrics)
        
        results_df = pd.DataFrame(results)
        return SensitivityResult(
            analysis_type="Outlier Impact",
            results_df=results_df,
            metrics={'total_outliers': results_df['outliers_pct'].sum()}
        )

    @staticmethod
    def analyze_transformation_impact(data: pd.DataFrame,
                                   variables: List[str],
                                   transformations: List[str],
                                   metrics: List[str]) -> SensitivityResult:
        """Analyze the impact of different variable transformations."""
        results = []
        original_data = data[variables].copy()
        
        for var in variables:
            var_data = original_data[var]
            
            for transform in transformations:
                try:
                    # Apply transformation
                    if transform == "Log":
                        transformed = pd.Series(
                            np.log(var_data - var_data.min() + 1),
                            name=var
                        )
                    elif transform == "Square Root":
                        transformed = pd.Series(
                            np.sqrt(var_data - var_data.min()),
                            name=var
                        )
                    elif transform == "Box-Cox":
                        transformed_data, _ = stats.boxcox(var_data - var_data.min() + 1)
                        transformed = pd.Series(transformed_data, name=var)
                    elif transform == "Yeo-Johnson":
                        transformed_data, _ = stats.yeojohnson(var_data)
                        transformed = pd.Series(transformed_data, name=var)
                    
                    # Calculate metrics
                    result = {
                        'variable': var,
                        'transformation': transform,
                        'original_skew': var_data.skew(),
                        'transformed_skew': transformed.skew(),
                        'original_kurt': var_data.kurtosis(),
                        'transformed_kurt': transformed.kurtosis()
                    }
                    
                    if "Normality" in metrics:
                        _, p_value = stats.normaltest(transformed)
                        result['normality_p_value'] = p_value
                    
                    results.append(result)
                
                except Exception as e:
                    st.warning(f"Could not apply {transform} to {var}: {str(e)}")
                    continue
        
        results_df = pd.DataFrame(results)
        return SensitivityResult(
            analysis_type="Variable Transformation",
            results_df=results_df,
            metrics={'transformations_applied': len(results)}
        )

    @staticmethod
    def analyze_sample_size_impact(data: pd.DataFrame,
                                 variables: List[str],
                                 sample_sizes: tuple,
                                 n_samples: int) -> SensitivityResult:
        """Analyze the impact of different sample sizes on statistical measures."""
        results = []
        original_data = data[variables].copy()
        
        for size_pct in range(sample_sizes[0], sample_sizes[1] + 1, 10):
            sample_size = int(len(data) * size_pct / 100)
            
            for _ in range(n_samples):
                # Random sampling
                sample = original_data.sample(n=sample_size, replace=False)
                
                # Calculate statistics
                metrics = {
                    'sample_size_pct': size_pct,
                    'actual_size': sample_size
                }
                
                # Calculate metrics for each variable
                for var in variables:
                    metrics.update({
                        f'{var}_mean': sample[var].mean(),
                        f'{var}_std': sample[var].std(),
                        f'{var}_skew': sample[var].skew(),
                        f'{var}_kurt': sample[var].kurtosis()
                    })
                
                results.append(metrics)
        
        results_df = pd.DataFrame(results)
        return SensitivityResult(
            analysis_type="Sample Size Impact",
            results_df=results_df,
            metrics={'total_samples': len(results)}
        )

    @staticmethod
    def _impute_missing_values(data: pd.DataFrame, method: str) -> pd.DataFrame:
        """Impute missing values using the specified method."""
        imputed_data = data.copy()
        
        for column in data.columns:
            if method == "Mean":
                imputed_data[column].fillna(data[column].mean(), inplace=True)
            elif method == "Median":
                imputed_data[column].fillna(data[column].median(), inplace=True)
            elif method == "Mode":
                imputed_data[column].fillna(data[column].mode()[0], inplace=True)
            elif method == "Linear Interpolation":
                imputed_data[column].interpolate(method='linear', inplace=True)
        
        return imputed_data

    @staticmethod
    def create_sensitivity_plots(result: SensitivityResult):
        """Create plots for sensitivity analysis results."""
        if result.analysis_type == "Missing Data Impact":
            fig = px.line(
                result.results_df,
                x='percentage',
                y=['mean_diff', 'std_diff', 'corr'],
                color='method',
                title='Impact of Missing Data by Imputation Method'
            )
            st.plotly_chart(fig)
        
        elif result.analysis_type == "Variable Transformation":
            metrics_to_plot = ['skew', 'kurt']
            for metric in metrics_to_plot:
                fig = px.bar(
                    result.results_df,
                    x='variable',
                    y=[f'original_{metric}', f'transformed_{metric}'],
                    barmode='group',
                    facet_col='transformation',
                    title=f'Impact of Transformations on {metric.capitalize()}'
                )
                st.plotly_chart(fig)
            
            if 'normality_p_value' in result.results_df.columns:
                fig = px.scatter(
                    result.results_df,
                    x='variable',
                    y='normality_p_value',
                    color='transformation',
                    title='Normality Test P-values by Transformation'
                )
                fig.add_hline(y=0.05, line_dash="dash", annotation_text="α = 0.05")
                st.plotly_chart(fig)
        
        elif result.analysis_type == "Outlier Impact":
            fig = px.bar(
                result.results_df,
                x='variable',
                y=['original_mean', 'treated_mean'],
                color='treatment',
                facet_col='method',
                barmode='group',
                title='Impact of Outlier Treatment on Mean'
            )
            st.plotly_chart(fig)
            
            fig = px.scatter(
                result.results_df,
                x='variable',
                y='outliers_pct',
                color='method',
                title='Percentage of Outliers Detected by Method'
            )
            st.plotly_chart(fig)
        
        elif result.analysis_type == "Sample Size Impact":
            metrics = [col for col in result.results_df.columns 
                      if any(x in col for x in ['mean', 'std', 'skew', 'kurt'])]
            melted = pd.melt(
                result.results_df,
                id_vars=['sample_size_pct', 'actual_size'],
                value_vars=metrics,
                var_name='metric',
                value_name='value'
            )
            
            fig = px.line(
                melted,
                x='sample_size_pct',
                y='value',
                color='metric',
                facet_col='metric',
                title='Impact of Sample Size on Statistical Measures'
            )
            st.plotly_chart(fig) 