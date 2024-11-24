from dataclasses import dataclass
from enum import Enum
from typing import Optional
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import pingouin as pg

print(f"Pingouin version: {pg.__version__}")

class MediationMethod(Enum):
    STATSMODELS = "statsmodels"
    PINGOUIN = "pingouin"

@dataclass
class MediationResult:
    total_effect: float
    direct_effect: float
    indirect_effect: float
    total_effect_p: float
    direct_effect_p: float
    indirect_effect_ci: tuple
    proportion_mediated: float
    sobel_statistic: float
    sobel_p: float
    independent_var: str
    method: str

class MediationAnalyzer:
    @staticmethod
    def perform_multiple_mediations(data: pd.DataFrame,
                                  independent_vars: list[str],
                                  mediator: str,
                                  dependent_var: str,
                                  method: MediationMethod = MediationMethod.STATSMODELS,
                                  confidence_level: float = 0.95,
                                  n_bootstrap: int = 5000) -> dict[str, MediationResult]:
        """Perform mediation analysis for multiple independent variables."""
        results = {}
        for iv in independent_vars:
            results[iv] = MediationAnalyzer.perform_mediation(
                data=data,
                independent_var=iv,
                mediator=mediator,
                dependent_var=dependent_var,
                method=method,
                confidence_level=confidence_level,
                n_bootstrap=n_bootstrap
            )
        return results

    @staticmethod
    def perform_mediation(data: pd.DataFrame,
                         independent_var: str,
                         mediator: str,
                         dependent_var: str,
                         method: MediationMethod = MediationMethod.STATSMODELS,
                         confidence_level: float = 0.95,
                         n_bootstrap: int = 5000) -> MediationResult:
        """Perform mediation analysis using the specified method."""
        
        # Input validation
        if not all(col in data.columns for col in [independent_var, mediator, dependent_var]):
            raise ValueError("One or more specified variables not found in the dataset")
            
        if method == MediationMethod.PINGOUIN:
            return MediationAnalyzer._perform_pingouin_mediation(
                data, independent_var, mediator, dependent_var, confidence_level, n_bootstrap)
        else:
            return MediationAnalyzer._perform_statsmodels_mediation(
                data, independent_var, mediator, dependent_var, confidence_level, n_bootstrap)

    @staticmethod
    def _perform_statsmodels_mediation(data: pd.DataFrame,
                                      independent_var: str,
                                      mediator: str,
                                      dependent_var: str,
                                      confidence_level: float,
                                      n_bootstrap: int) -> MediationResult:
        """Perform mediation analysis using statsmodels."""
        # Path c (total effect)
        X = sm.add_constant(data[independent_var])
        model_c = sm.OLS(data[dependent_var], X).fit()
        c = model_c.params[1]
        c_p = model_c.pvalues[1]
        
        # Path a
        model_a = sm.OLS(data[mediator], X).fit()
        a = model_a.params[1]
        
        # Path b
        X_prime = sm.add_constant(pd.DataFrame({
            independent_var: data[independent_var],
            mediator: data[mediator]
        }))
        model_b = sm.OLS(data[dependent_var], X_prime).fit()
        b = model_b.params[2]
        c_prime = model_b.params[1]  # direct effect
        
        # Indirect effect
        ab = a * b
        
        # Bootstrap confidence interval
        indirect_effects = []
        for _ in range(n_bootstrap):
            indices = np.random.randint(0, len(data), len(data))
            boot_data = data.iloc[indices]
            
            # Recalculate paths
            X_boot = sm.add_constant(boot_data[independent_var])
            a_boot = sm.OLS(boot_data[mediator], X_boot).fit().params[1]
            
            X_prime_boot = sm.add_constant(pd.DataFrame({
                independent_var: boot_data[independent_var],
                mediator: boot_data[mediator]
            }))
            b_boot = sm.OLS(boot_data[dependent_var], X_prime_boot).fit().params[2]
            
            indirect_effects.append(a_boot * b_boot)
        
        # Calculate confidence interval
        ci_lower = np.percentile(indirect_effects, (1 - confidence_level) * 100 / 2)
        ci_upper = np.percentile(indirect_effects, 100 - (1 - confidence_level) * 100 / 2)
        
        # Sobel test
        a_se = model_a.bse[1]
        b_se = model_b.bse[2]
        sobel_se = np.sqrt(b**2 * a_se**2 + a**2 * b_se**2)
        sobel_z = ab / sobel_se
        sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))
        
        return MediationResult(
            total_effect=c,
            direct_effect=c_prime,
            indirect_effect=ab,
            total_effect_p=c_p,
            direct_effect_p=model_b.pvalues[1],
            indirect_effect_ci=(ci_lower, ci_upper),
            proportion_mediated=ab/c if c != 0 else 0,
            sobel_statistic=sobel_z,
            sobel_p=sobel_p,
            independent_var=independent_var,
            method='statsmodels'
        )

    @staticmethod
    def _perform_pingouin_mediation(data: pd.DataFrame,
                                  independent_var: str,
                                  mediator: str,
                                  dependent_var: str,
                                  confidence_level: float,
                                  n_bootstrap: int) -> MediationResult:
        """Perform mediation analysis using pingouin."""
        try:
            # Perform mediation analysis
            stats = pg.mediation_analysis(
                data=data,
                x=independent_var,
                m=mediator,
                y=dependent_var,
                n_boot=n_bootstrap,
                alpha=(1 - confidence_level)
            )
            
            # Debug print
            print("\nPingouin Results Structure:")
            print(stats)
            print("\nShape:", stats.shape)
            print("\nColumns:", stats.columns.tolist())
            
            try:
                # Extract values more safely
                # Get the row containing the indirect effect (should be labeled 'Indirect')
                indirect_row = stats[stats['path'].str.contains('Indirect', case=False, na=False)]
                direct_row = stats[stats['path'].str.contains('Direct', case=False, na=False)]
                total_row = stats[stats['path'].str.contains('Total', case=False, na=False)]
                
                # Extract effects
                indirect_effect = float(indirect_row['coef'].iloc[0])
                direct_effect = float(direct_row['coef'].iloc[0])
                total_effect = float(total_row['coef'].iloc[0])
                
                # Extract p-values
                indirect_p = float(indirect_row['pval'].iloc[0])
                direct_p = float(direct_row['pval'].iloc[0])
                total_p = float(total_row['pval'].iloc[0])
                
                # Extract confidence intervals
                ci_lower = float(indirect_row['CI[2.5%]'].iloc[0])
                ci_upper = float(indirect_row['CI[97.5%]'].iloc[0])
                
                # Calculate proportion mediated
                prop_mediated = indirect_effect / total_effect if total_effect != 0 else 0
                
                # Get standard error for Sobel test
                se = float(indirect_row['se'].iloc[0])
                sobel_z = indirect_effect / se if se != 0 else 0
                
                return MediationResult(
                    total_effect=total_effect,
                    direct_effect=direct_effect,
                    indirect_effect=indirect_effect,
                    total_effect_p=total_p,
                    direct_effect_p=direct_p,
                    indirect_effect_ci=(ci_lower, ci_upper),
                    proportion_mediated=prop_mediated,
                    sobel_statistic=sobel_z,
                    sobel_p=indirect_p,
                    independent_var=independent_var,
                    method='pingouin'
                )
                
            except Exception as e:
                print(f"\nError extracting values: {str(e)}")
                print("\nFull stats DataFrame:")
                print(stats.to_string())
                raise ValueError(f"Unable to extract required statistics: {str(e)}")
                
        except Exception as e:
            print(f"\nException in Pingouin analysis: {str(e)}")
            print(f"Exception type: {type(e)}")
            raise ValueError(f"Error during Pingouin mediation analysis: {str(e)}")
