import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Any
from config.settings import SIGNIFICANCE_LEVEL

@dataclass
class StatTestResult:
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    groups: Dict[str, Any]
    additional_info: Dict = None
    target: str = None
    group_var: str = None

class StatisticalAnalyzer:
    @staticmethod
    def perform_two_group_test(data: pd.DataFrame,
                             target: str,
                             group: str) -> StatTestResult:
        """Perform t-test or Mann-Whitney U test based on normality."""
        groups = data[group].unique()
        if len(groups) != 2:
            raise ValueError("Two-group test requires exactly 2 groups")
        
        group1_data = data[data[group] == groups[0]][target]
        group2_data = data[data[group] == groups[1]][target]
        
        # Check normality
        _, p_norm1 = stats.shapiro(group1_data)
        _, p_norm2 = stats.shapiro(group2_data)
        is_normal = p_norm1 > SIGNIFICANCE_LEVEL and p_norm2 > SIGNIFICANCE_LEVEL
        
        if is_normal:
            stat, p_value = stats.ttest_ind(group1_data, group2_data)
            test_name = "Independent t-test"
        else:
            stat, p_value = stats.mannwhitneyu(group1_data, group2_data)
            test_name = "Mann-Whitney U test"
        
        return StatTestResult(
            test_name=test_name,
            statistic=stat,
            p_value=p_value,
            significant=p_value < SIGNIFICANCE_LEVEL,
            groups={str(groups[0]): group1_data, str(groups[1]): group2_data},
            additional_info={"normality_assumption": is_normal}
        )

    @staticmethod
    def perform_multi_group_test(data: pd.DataFrame,
                               target: str,
                               group: str) -> StatTestResult:
        """Perform ANOVA or Kruskal-Wallis test based on normality."""
        groups = {}
        for name, group_data in data.groupby(group)[target]:
            groups[str(name)] = group_data
        
        # Check normality for all groups
        is_normal = all(stats.shapiro(group)[1] > SIGNIFICANCE_LEVEL 
                       for group in groups.values())
        
        if is_normal:
            stat, p_value = stats.f_oneway(*groups.values())
            test_name = "One-way ANOVA"
        else:
            stat, p_value = stats.kruskal(*groups.values())
            test_name = "Kruskal-Wallis H-test"
        
        return StatTestResult(
            test_name=test_name,
            statistic=stat,
            p_value=p_value,
            significant=p_value < SIGNIFICANCE_LEVEL,
            groups=groups,
            additional_info={"normality_assumption": is_normal}
        )

    @staticmethod
    def perform_chi_square_test(data: pd.DataFrame,
                              var1: str,
                              var2: str) -> StatTestResult:
        """Perform chi-square test of independence."""
        contingency = pd.crosstab(data[var1], data[var2])
        stat, p_value, dof, expected = stats.chi2_contingency(contingency)
        
        return StatTestResult(
            test_name="Chi-square test of independence",
            statistic=stat,
            p_value=p_value,
            significant=p_value < SIGNIFICANCE_LEVEL,
            groups={"contingency": contingency, "expected": expected},
            additional_info={"degrees_of_freedom": dof}
        )

    @staticmethod
    def batch_two_group_tests(data: pd.DataFrame,
                             targets: List[str],
                             groups: List[str]) -> List[StatTestResult]:
        """Perform two-group tests for multiple targets and grouping variables."""
        results = []
        for target in targets:
            for group in groups:
                try:
                    result = StatisticalAnalyzer.perform_two_group_test(
                        data=data,
                        target=target,
                        group=group
                    )
                    result.target = target
                    result.group_var = group
                    results.append(result)
                except ValueError as e:
                    # Skip combinations that don't meet requirements
                    continue
        return results

    @staticmethod
    def batch_multi_group_tests(data: pd.DataFrame,
                               targets: List[str],
                               groups: List[str]) -> List[StatTestResult]:
        """Perform multi-group tests for multiple targets and grouping variables."""
        results = []
        for target in targets:
            for group in groups:
                try:
                    result = StatisticalAnalyzer.perform_multi_group_test(
                        data=data,
                        target=target,
                        group=group
                    )
                    result.target = target
                    result.group_var = group
                    results.append(result)
                except ValueError as e:
                    # Skip combinations that don't meet requirements
                    continue
        return results

    @staticmethod
    def batch_chi_square_tests(data: pd.DataFrame,
                              vars1: List[str],
                              vars2: List[str]) -> List[StatTestResult]:
        """Perform chi-square tests for multiple variable combinations."""
        results = []
        for var1 in vars1:
            for var2 in vars2:
                if var1 != var2:  # Avoid testing variable against itself
                    try:
                        result = StatisticalAnalyzer.perform_chi_square_test(
                            data=data,
                            var1=var1,
                            var2=var2
                        )
                        result.target = var1
                        result.group_var = var2
                        results.append(result)
                    except ValueError as e:
                        # Skip combinations that cause errors
                        continue
        return results