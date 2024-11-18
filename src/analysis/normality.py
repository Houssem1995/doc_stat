import pandas as pd
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Optional
from config.settings import SIGNIFICANCE_LEVEL

@dataclass
class NormalityTestResult:
    variable: str
    shapiro_stat: Optional[float] = None
    shapiro_p: Optional[float] = None
    dagostino_stat: Optional[float] = None
    dagostino_p: Optional[float] = None
    is_normal: bool = False
    sample_size: int = 0
    skewness: float = 0.0
    kurtosis: float = 0.0
    missing_values: int = 0
    mean: float = 0.0
    median: float = 0.0
    std_dev: float = 0.0
    iqr: float = 0.0
    recommendation: str = ""
    test_method: str = ""

class NormalityAnalyzer:
    @staticmethod
    def analyze_multiple(data: pd.DataFrame, 
                        columns: List[str], 
                        significance_level: float = 0.05,
                        test_method: str = "Both") -> Dict[str, NormalityTestResult]:
        """Perform normality tests on multiple columns."""
        results = {}
        for column in columns:
            clean_data = data[column].dropna()
            
            # Basic statistics
            skew = stats.skew(clean_data)
            kurt = stats.kurtosis(clean_data)
            missing = data[column].isna().sum()
            
            # Additional statistics
            mean = np.mean(clean_data)
            median = np.median(clean_data)
            std_dev = np.std(clean_data)
            q75, q25 = np.percentile(clean_data, [75, 25])
            iqr = q75 - q25
            
            # Initialize test results
            shapiro_stat = shapiro_p = dagostino_stat = dagostino_p = None
            
            # Perform selected tests
            if test_method in ["Shapiro-Wilk", "Both"]:
                shapiro_stat, shapiro_p = stats.shapiro(clean_data)
            
            if test_method in ["D'Agostino-Pearson", "Both"]:
                dagostino_stat, dagostino_p = stats.normaltest(clean_data)
            
            # Determine normality based on selected test
            is_normal = False
            if test_method == "Shapiro-Wilk":
                is_normal = shapiro_p > significance_level
            elif test_method == "D'Agostino-Pearson":
                is_normal = dagostino_p > significance_level
            else:  # Both
                is_normal = (shapiro_p > significance_level and 
                           dagostino_p > significance_level)
            
            # Generate recommendation
            recommendation = NormalityAnalyzer._generate_detailed_recommendation(
                is_normal=is_normal,
                skewness=skew,
                kurtosis=kurt,
                sample_size=len(clean_data),
                shapiro_p=shapiro_p,
                dagostino_p=dagostino_p,
                significance_level=significance_level,
                test_method=test_method
            )
            
            results[column] = NormalityTestResult(
                variable=column,
                shapiro_stat=shapiro_stat,
                shapiro_p=shapiro_p,
                dagostino_stat=dagostino_stat,
                dagostino_p=dagostino_p,
                is_normal=is_normal,
                sample_size=len(clean_data),
                skewness=skew,
                kurtosis=kurt,
                missing_values=missing,
                mean=mean,
                median=median,
                std_dev=std_dev,
                iqr=iqr,
                recommendation=recommendation,
                test_method=test_method
            )
        
        return results

    @staticmethod
    def _generate_detailed_recommendation(is_normal: bool, 
                                        skewness: float, 
                                        kurtosis: float, 
                                        sample_size: int,
                                        shapiro_p: float,
                                        dagostino_p: float,
                                        significance_level: float,
                                        test_method: str) -> str:
        """Generate detailed recommendations based on test results."""
        recommendations = []
        
        if is_normal:
            recommendations.append("✅ The data appears to be normally distributed.")
            recommendations.append("→ Parametric tests can be used (e.g., t-tests, ANOVA).")
        else:
            recommendations.append("❌ The data does not follow a normal distribution.")
            recommendations.append("→ Consider using non-parametric alternatives.")
            
            # Add specific recommendations based on distribution characteristics
            if abs(skewness) > 1:
                if skewness > 0:
                    recommendations.append("→ Data is right-skewed. Consider log transformation.")
                else:
                    recommendations.append("→ Data is left-skewed. Consider exponential transformation.")
            
            if abs(kurtosis) > 1:
                if kurtosis > 0:
                    recommendations.append("→ Data has heavy tails. Consider robust statistical methods.")
                else:
                    recommendations.append("→ Data has light tails.")
        
        if sample_size < 30:
            recommendations.append("⚠️ Small sample size - interpret results with caution.")
        
        return "\n".join(recommendations)

    @staticmethod
    def create_summary_table(results: Dict[str, NormalityTestResult], 
                           test_method: str) -> pd.DataFrame:
        """Create a summary table of normality test results."""
        summary_data = []
        for result in results.values():
            data = {
                'Variable': result.variable,
                'Sample Size': result.sample_size,
                'Missing Values': result.missing_values,
                'Skewness': f"{result.skewness:.2f}",
                'Kurtosis': f"{result.kurtosis:.2f}",
                'Normal': '✅' if result.is_normal else '❌'
            }
            
            # Add test-specific results
            if test_method in ["Shapiro-Wilk", "Both"]:
                data['Shapiro p-value'] = f"{result.shapiro_p:.4f}"
            
            if test_method in ["D'Agostino-Pearson", "Both"]:
                data["D'Agostino p-value"] = f"{result.dagostino_p:.4f}"
            
            summary_data.append(data)
        
        return pd.DataFrame(summary_data)

    @staticmethod
    def create_detailed_stats_table(results: Dict[str, NormalityTestResult]) -> pd.DataFrame:
        """Create a detailed statistics table."""
        detailed_data = []
        for result in results.values():
            detailed_data.append({
                'Variable': result.variable,
                'Mean': f"{result.mean:.4f}",
                'Median': f"{result.median:.4f}",
                'Std Dev': f"{result.std_dev:.4f}",
                'IQR': f"{result.iqr:.4f}",
                'Skewness': f"{result.skewness:.4f}",
                'Kurtosis': f"{result.kurtosis:.4f}",
                'Shapiro-Wilk Stat': f"{result.shapiro_stat:.4f}",
                "D'Agostino Stat": f"{result.dagostino_stat:.4f}",
                'Sample Size': result.sample_size,
                'Missing Values': result.missing_values
            })
        
        return pd.DataFrame(detailed_data)