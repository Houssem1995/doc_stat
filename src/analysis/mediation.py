from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

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

class MediationAnalyzer:
    @staticmethod
    def perform_mediation(data: pd.DataFrame,
                         independent_var: str,
                         mediator: str,
                         dependent_var: str,
                         confidence_level: float = 0.95,
                         n_bootstrap: int = 5000) -> MediationResult:
        """Perform mediation analysis using bootstrap method."""
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
            sobel_p=sobel_p
        ) 