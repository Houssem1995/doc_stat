import streamlit as st
import pandas as pd
from src.data.data_loader import DataLoader
from src.analysis.normality import NormalityAnalyzer
from src.analysis.correlation import CorrelationAnalyzer
from src.analysis.statistical import StatisticalAnalyzer
from src.visualization.normality_plots import create_multiple_normality_plots
from src.visualization.correlation_plots import create_correlation_plots
from src.visualization.statistical_plots import create_statistical_plots
from config.settings import TEST_TYPES
from src.analysis.regression import RegressionAnalyzer
from src.analysis.mediation import MediationAnalyzer, MediationMethod, MediationResult
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import io
from openpyxl.utils import get_column_letter
from src.analysis.sensitivity import SensitivityAnalyzer
from src.analysis.synthetic_control import SyntheticControlAnalyzer


class DocStatApp:
    def __init__(self):
        st.set_page_config(page_title="DocStat", layout="wide")
        if "data" not in st.session_state:
            st.session_state.data = None
        if "categorical_numeric_cols" not in st.session_state:
            st.session_state.categorical_numeric_cols = []

    def get_numeric_columns(self) -> list:
        """Get list of numeric columns from the loaded dataset."""
        if st.session_state.data is not None:
            numeric_cols = st.session_state.data.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()
            # Exclude columns marked as categorical
            return [
                col
                for col in numeric_cols
                if col not in st.session_state.categorical_numeric_cols
            ]
        return []

    def get_categorical_columns(self) -> list:
        """Get list of categorical columns from the loaded dataset."""
        if st.session_state.data is not None:
            # Include both native categorical and user-specified categorical columns
            native_categorical = st.session_state.data.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
            return native_categorical + st.session_state.categorical_numeric_cols
        return []

    def _render_correlation_tab(self):
        """Render the correlation analysis tab."""
        st.header("Correlation Analysis")

        if st.session_state.data is None:
            st.warning("Please upload a dataset first.")
            return

        # Get numeric columns
        numeric_cols = self.get_numeric_columns()

        if not numeric_cols:
            st.warning("No numeric columns found in the dataset.")
            return

        # Column selection
        selected_columns = st.multiselect(
            "Select Variables for Correlation Analysis",
            options=numeric_cols,
            help="Select two or more variables",
            key="correlation_vars",
        )

        # Method selection
        col1, col2 = st.columns(2)

        with col1:
            method = st.radio(
                "Select Correlation Method",
                options=["pearson", "spearman"],
                help="""
                - Pearson: Linear relationships (requires normal distribution)
                - Spearman: Monotonic relationships (no normality assumption)
                """,
                key="correlation_method",
            )

        with col2:
            significance_level = st.slider(
                "Significance Level (α)",
                min_value=0.01,
                max_value=0.10,
                value=0.05,
                step=0.01,
                key="correlation_significance",
                help="Threshold for statistical significance",
            )

        if (
            st.button("Perform Correlation Analysis", key="run_correlation")
            and len(selected_columns) >= 2
        ):
            try:
                # Perform correlation analysis
                results = CorrelationAnalyzer.calculate_correlation(
                    data=st.session_state.data,
                    variables=selected_columns,
                    method=method,
                    significance_level=significance_level,
                )

                # Display correlation matrix
                st.write("### Correlation Matrix")
                correlation_df = results.correlation_matrix.style.background_gradient(
                    cmap="RdYlBu", vmin=-1, vmax=1
                ).format("{:.3f}")
                st.dataframe(correlation_df)

                # Display p-values
                st.write("### P-Values Matrix")
                pvalue_df = results.p_value_matrix.style.background_gradient(
                    cmap="Reds_r"
                ).format("{:.3f}")
                st.dataframe(pvalue_df)

                # Create correlation heatmap
                st.write("### Correlation Heatmap")
                create_correlation_plots(results)

                # Display significant correlations
                if results.significant_pairs:
                    st.write("### Significant Correlations")
                    significant_df = pd.DataFrame(
                        results.significant_pairs,
                        columns=["Variable 1", "Variable 2", "Correlation"],
                    ).sort_values("Correlation", key=abs, ascending=False)

                    # Add significance indicators
                    significant_df["Strength"] = significant_df["Correlation"].apply(
                        lambda x: (
                            "🔴 Strong"
                            if abs(x) > 0.7
                            else "🟡 Moderate" if abs(x) > 0.3 else "🟢 Weak"
                        )
                    )

                    st.dataframe(significant_df.round(3))

                # Export options
                col1, col2 = st.columns(2)
                with col1:
                    csv_corr = results.correlation_matrix.to_csv(index=True)
                    st.download_button(
                        label="Download Correlation Matrix",
                        data=csv_corr,
                        file_name="correlation_matrix.csv",
                        mime="text/csv",
                        key="download_correlation_matrix",
                    )
                with col2:
                    csv_pval = results.p_value_matrix.to_csv(index=True)
                    st.download_button(
                        label="Download P-Values Matrix",
                        data=csv_pval,
                        file_name="p_values_matrix.csv",
                        mime="text/csv",
                        key="download_pvalues_matrix",
                    )

            except Exception as e:
                st.error(f"Error during correlation analysis: {str(e)}")
                st.exception(e)
        elif len(selected_columns) < 2 and st.button("Perform Correlation Analysis"):
            st.warning("Please select at least two variables for correlation analysis.")

    def _render_normality_tab(self):
        st.header("Normality Analysis")

        # Add explanation of normality tests
        with st.expander("ℹ️ About Normality Tests", expanded=False):
            st.markdown(
                """
            ### Available Normality Tests
            
            1. **Shapiro-Wilk Test**
               - Best for sample sizes < 2000
               - Most powerful normality test
               - Null hypothesis: data is normally distributed
               - Recommended for small to medium datasets
            
            2. **D'Agostino-Pearson Test**
               - Suitable for larger samples
               - Tests both skewness and kurtosis
               - More robust for larger datasets
               - Recommended for samples > 2000
            
            ### Test Selection Guidelines:
            - For n < 2000: Use Shapiro-Wilk
            - For n ≥ 2000: Use D'Agostino-Pearson
            - For comprehensive analysis: Use both
            """
            )

        if st.session_state.data is None:
            st.warning("Please upload a dataset first.")
            return

        # Get numeric columns
        numeric_cols = self.get_numeric_columns()

        if not numeric_cols:
            st.warning("No numeric columns found in the dataset.")
            return

        # Create two columns for settings
        col1, col2 = st.columns(2)

        with col1:
            # Multiple column selection
            selected_columns = st.multiselect(
                "Select Variables for Normality Testing",
                options=numeric_cols,
                help="You can select multiple variables for analysis",
                key="normality_column_select",
            )

        with col2:
            # Test selection
            test_method = st.radio(
                "Select Normality Test",
                options=["Shapiro-Wilk", "D'Agostino-Pearson", "Both"],
                help="Choose the normality test method",
                key="normality_test_method",
            )

        # Add test configuration options
        with st.expander("⚙️ Test Configuration", expanded=False):
            significance_level = st.slider(
                "Significance Level (α)",
                min_value=0.01,
                max_value=0.10,
                value=0.05,
                step=0.01,
                key="significance_level",
                help="Probability threshold for determining statistical significance",
            )

            show_detailed_stats = st.checkbox(
                "Show Detailed Statistics",
                value=False,
                key="show_detailed_stats",
                help="Display additional statistical measures",
            )

        if (
            st.button("Perform Normality Tests", key="run_normality")
            and selected_columns
        ):
            try:
                # Perform analysis
                results = NormalityAnalyzer.analyze_multiple(
                    data=st.session_state.data,
                    columns=selected_columns,
                    significance_level=significance_level,
                    test_method=test_method,
                )

                # Display summary table
                st.write("### Summary of Normality Tests")
                summary_df = NormalityAnalyzer.create_summary_table(
                    results, test_method
                )
                st.dataframe(summary_df, use_container_width=True)

                # Display detailed statistics if requested
                if show_detailed_stats:
                    st.write("### Detailed Statistics")
                    detailed_df = NormalityAnalyzer.create_detailed_stats_table(results)
                    st.dataframe(detailed_df, use_container_width=True)

                # Display detailed analysis for each variable
                st.write("### Detailed Analysis")
                create_multiple_normality_plots(st.session_state.data, results)

                # Export options
                col1, col2 = st.columns(2)
                with col1:
                    csv_summary = summary_df.to_csv(index=False)
                    st.download_button(
                        label="Download Summary as CSV",
                        data=csv_summary,
                        file_name="normality_test_summary.csv",
                        mime="text/csv",
                        key="download_normality_summary",
                    )

                if show_detailed_stats:
                    with col2:
                        csv_detailed = detailed_df.to_csv(index=False)
                        st.download_button(
                            label="Download Detailed Statistics",
                            data=csv_detailed,
                            file_name="normality_test_detailed.csv",
                            mime="text/csv",
                            key="download_normality_detailed",
                        )

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.exception(e)

    def _render_statistical_tab(self):
        st.header("Statistical Tests")

        if st.session_state.data is None:
            st.warning("Please upload a dataset first.")
            return

        # Get columns
        numeric_cols = self.get_numeric_columns()
        categorical_cols = self.get_categorical_columns()

        if not numeric_cols or not categorical_cols:
            st.warning(
                "Dataset must contain both numeric and categorical columns for statistical tests."
            )
            return

        # Test type selection
        test_type = st.radio(
            "Select Test Type",
            options=[
                "Two Groups (T-test/Mann-Whitney)",
                "Multiple Groups (ANOVA/Kruskal)",
                "Categorical (Chi-square)",
            ],
            help="Select the appropriate test based on your data",
            key="statistical_test_type",
        )

        # Variable selection based on test type
        if "Chi-square" in test_type:
            col1, col2 = st.columns(2)
            with col1:
                vars1 = st.multiselect(
                    "Select First Set of Variables",
                    categorical_cols,
                    help="Select one or more categorical variables",
                    key="stat_vars1",
                )

            with col2:
                vars2 = st.multiselect(
                    "Select Second Set of Variables",
                    categorical_cols,
                    help="Select one or more categorical variables",
                    key="stat_vars2",
                )
        else:
            col1, col2 = st.columns(2)
            with col1:
                target_vars = st.multiselect(
                    "Select Dependent Variable(s)",
                    numeric_cols,
                    help="Select one or more numeric variables to analyze",
                    key="stat_targets",
                )

            with col2:
                group_vars = st.multiselect(
                    "Select Grouping Variable(s)",
                    categorical_cols,
                    help="Select one or more categorical variables for grouping",
                    key="stat_groups",
                )

        if st.button("Perform Statistical Tests", key="run_statistical"):
            try:
                if "Chi-square" in test_type:
                    if not vars1 or not vars2:
                        st.warning("Please select variables for both sets.")
                        return

                    results = StatisticalAnalyzer.batch_chi_square_tests(
                        st.session_state.data, vars1, vars2
                    )
                elif "Two Groups" in test_type:
                    if not target_vars or not group_vars:
                        st.warning("Please select both target and grouping variables.")
                        return

                    results = StatisticalAnalyzer.batch_two_group_tests(
                        st.session_state.data, target_vars, group_vars
                    )
                else:  # Multiple Groups
                    if not target_vars or not group_vars:
                        st.warning("Please select both target and grouping variables.")
                        return

                    results = StatisticalAnalyzer.batch_multi_group_tests(
                        st.session_state.data, target_vars, group_vars
                    )

                # Display results
                if not results:
                    st.warning(
                        "No valid test results were produced. Check your variable selections."
                    )
                    return

                for result in results:
                    with st.expander(
                        f"Results for {result.target} by {result.group_var}",
                        expanded=True,
                    ):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.write("### Test Results")
                            st.write(f"**Test**: {result.test_name}")
                            st.write(f"**Statistic**: {result.statistic:.4f}")
                            st.write(f"**P-value**: {result.p_value:.4f}")

                            if result.significant:
                                st.success("✅ Significant difference detected")
                            else:
                                st.info("ℹ️ No significant difference detected")

                        with col2:
                            st.write("### Group Statistics")
                            if isinstance(result.groups, dict):
                                if "contingency" in result.groups:
                                    # For chi-square tests
                                    st.write("Contingency Table:")
                                    st.dataframe(result.groups["contingency"])
                                else:
                                    # For other tests
                                    st.write("Group Statistics:")
                                    st.dataframe(result.groups)
                            else:
                                # For other tests
                                st.write("Group Statistics:")
                                st.dataframe(result.groups)
            except Exception as e:
                st.error(f"Error during statistical analysis: {str(e)}")
                st.exception(e)

    def _render_regression_tab(self):
        st.header("Regression Analysis")

        if st.session_state.data is None:
            st.warning("Please upload a dataset first.")
            return

        # Get numeric columns
        numeric_cols = self.get_numeric_columns()

        if not numeric_cols:
            st.warning("No numeric columns found in the dataset.")
            return

        # Variable selection
        col1, col2 = st.columns(2)

        with col1:
            dependent_var = st.selectbox(
                "Select Dependent Variable (Y)",
                options=numeric_cols,
                key="reg_dependent",
            )

        with col2:
            independent_vars = st.multiselect(
                "Select Independent Variable(s) (X)",
                options=[col for col in numeric_cols if col != dependent_var],
                key="reg_independent",
            )

        # Analysis options
        with st.expander("Regression Options", expanded=False):
            test_size = st.slider(
                "Test Set Size",
                min_value=0.1,
                max_value=0.4,
                value=0.2,
                step=0.05,
                help="Proportion of data used for testing",
            )

            standardize = st.checkbox(
                "Standardize Variables",
                value=True,
                help="Convert variables to z-scores",
            )

        if st.button("Run Regression Analysis", key="run_regression"):
            if not independent_vars:
                st.warning("Please select at least one independent variable.")
                return

            try:
                results = RegressionAnalyzer.perform_regression(
                    data=st.session_state.data,
                    dependent_var=dependent_var,
                    independent_vars=independent_vars,
                    test_size=test_size,
                    standardize=standardize,
                )

                # Display results
                st.write("### Model Summary")
                st.write(f"R-squared: {results.r_squared:.3f}")
                st.write(f"Adjusted R-squared: {results.adjusted_r_squared:.3f}")
                st.write(f"RMSE: {results.rmse:.3f}")

                # Coefficients table
                st.write("### Coefficients")
                coef_df = pd.DataFrame(
                    {
                        "Coefficient": results.coefficients,
                        "Std Error": results.std_errors,
                        "P-value": results.p_values,
                    }
                ).round(4)
                st.dataframe(coef_df)

                # Plots
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### Residuals Plot")
                    fig_residuals = px.scatter(
                        x=results.predictions,
                        y=results.residuals,
                        labels={"x": "Predicted Values", "y": "Residuals"},
                    )
                    st.plotly_chart(fig_residuals)

                with col2:
                    st.write("### Q-Q Plot")
                    fig_qq = px.scatter(
                        x=np.sort(
                            stats.norm.ppf(
                                np.linspace(0.01, 0.99, len(results.residuals))
                            )
                        ),
                        y=np.sort(results.residuals),
                        labels={"x": "Theoretical Quantiles", "y": "Sample Quantiles"},
                    )
                    fig_qq.add_trace(
                        go.Scatter(
                            x=[-3, 3],
                            y=[-3, 3],
                            mode="lines",
                            line=dict(dash="dash"),
                            showlegend=False,
                        )
                    )
                    st.plotly_chart(fig_qq)

            except Exception as e:
                st.error(f"Error during regression analysis: {str(e)}")
                st.exception(e)

    def _render_mediation_tab(self):
        """Render the mediation analysis section with support for multiple mediators."""
        st.header("🔄 Mediation Analysis")

        if st.session_state.data is None:
            st.warning("Please upload a dataset first.")
            return

        # Method selection with detailed information
        method = st.selectbox(
            "Select Analysis Method",
            options=[method.value for method in MediationMethod],
            help="Choose which statistical library to use for the mediation analysis",
        )

        # Variable selection with support for multiple mediators
        col1, col2, col3 = st.columns(3)
        with col1:
            independent_vars = st.multiselect(
                "Independent Variable(s)",
                options=st.session_state.data.columns.tolist(),
                help="Select one or more independent variables",
            )

        with col2:
            mediators = st.multiselect(  # Changed from selectbox to multiselect
                "Mediator Variable(s)",  # Updated label
                options=[
                    col
                    for col in st.session_state.data.columns.tolist()
                    if col not in independent_vars
                ],
                help="Select one or more mediator variables",
            )

        with col3:
            dependent_var = st.selectbox(
                "Dependent Variable",
                options=[
                    col
                    for col in st.session_state.data.columns.tolist()
                    if col not in independent_vars and col not in mediators
                ],
                help="Select the dependent variable",
            )

        # Analysis parameters
        with st.expander("Advanced Settings", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                confidence_level = st.slider(
                    "Confidence Level",
                    min_value=0.8,
                    max_value=0.99,
                    value=0.95,
                    step=0.01,
                    help="Set the confidence level for the analysis",
                )

            with col2:
                n_bootstrap = st.number_input(
                    "Bootstrap Samples",
                    min_value=1000,
                    max_value=10000,
                    value=5000,
                    step=1000,
                    help="Set the number of bootstrap samples",
                )

        # Run analysis button
        if st.button("Run Mediation Analysis", type="primary"):
            if not independent_vars or not mediators or not dependent_var:
                st.warning("Please select all required variables.")
                return

            try:
                with st.spinner("Running mediation analysis..."):
                    results = MediationAnalyzer.perform_multiple_mediations(
                        data=st.session_state.data,
                        independent_vars=independent_vars,
                        mediators=mediators,
                        dependent_var=dependent_var,
                        method=MediationMethod(method),
                        confidence_level=confidence_level,
                        n_bootstrap=n_bootstrap,
                    )
                    self._display_mediation_results(independent_vars, results)

                    # Add download section
                    self._download_mediation_results(results)

            except Exception as e:
                st.error(f"An error occurred during the analysis: {str(e)}")
                st.exception(e)

    def _display_mediation_results(
        self,
        independent_vars: list[str],
        results: dict[str, dict[str, MediationResult]],
    ):
        """Display the mediation analysis results in a formatted way."""
        st.subheader("📊 Results")

        # Create tabs for each independent variable
        iv_tabs = st.tabs(independent_vars)

        for iv, iv_tab in zip(results.keys(), iv_tabs):
            with iv_tab:
                st.markdown(f"### Results for Independent Variable: {iv}")

                # Create expanders for each mediator
                for mediator, result in results[iv].items():
                    with st.expander(f"Mediator: {mediator}", expanded=True):
                        # Create three columns for organized display
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.markdown("#### Effect Sizes")
                            st.write(f"Total Effect (c): {result.total_effect:.4f}")
                            st.write(f"Direct Effect (c'): {result.direct_effect:.4f}")
                            st.write(
                                f"Indirect Effect (ab): {result.indirect_effect:.4f}"
                            )

                        with col2:
                            st.markdown("#### Statistical Tests")
                            st.write(
                                f"Total Effect p-value: {result.total_effect_p:.4f}"
                            )
                            st.write(
                                f"Direct Effect p-value: {result.direct_effect_p:.4f}"
                            )
                            st.write(f"Sobel Test p-value: {result.sobel_p:.4f}")

                        with col3:
                            st.markdown("#### Additional Metrics")
                            st.write(
                                f"Proportion Mediated: {result.proportion_mediated:.2%}"
                            )
                            st.write("Indirect Effect 95% CI:")
                            st.write(f"- Lower: {result.indirect_effect_ci[0]:.4f}")
                            st.write(f"- Upper: {result.indirect_effect_ci[1]:.4f}")

                        # Add significance indicators
                        if result.total_effect_p < 0.05:
                            st.success("✓ Significant total effect detected")
                        if (
                            result.indirect_effect_ci[0] * result.indirect_effect_ci[1]
                            > 0
                        ):
                            st.success("✓ Significant mediation effect detected")

    def _download_mediation_results(
        self, results: dict[str, dict[str, MediationResult]]
    ):
        """Create download buttons for results in different formats."""
        st.subheader("📥 Download Results")

        # Create DataFrame with all results
        records = []
        for iv, mediator_results in results.items():
            for mediator, result in mediator_results.items():
                record = {
                    "Independent Variable": iv,
                    "Mediator": mediator,
                    "Method": result.method,
                    "Total Effect": result.total_effect,
                    "Direct Effect": result.direct_effect,
                    "Indirect Effect": result.indirect_effect,
                    "Total Effect p-value": result.total_effect_p,
                    "Direct Effect p-value": result.direct_effect_p,
                    "Indirect Effect CI Lower": result.indirect_effect_ci[0],
                    "Indirect Effect CI Upper": result.indirect_effect_ci[1],
                    "Proportion Mediated": result.proportion_mediated,
                    "Sobel Statistic": result.sobel_statistic,
                    "Sobel p-value": result.sobel_p,
                }
                records.append(record)

        df_results = pd.DataFrame(records)

        col1, col2 = st.columns(2)

        # CSV download
        with col1:
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="mediation_analysis_results.csv",
                mime="text/csv",
                help="Download the results as a CSV file",
            )

        # Excel download
        with col2:
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df_results.to_excel(writer, sheet_name="Mediation Results", index=False)

                # Auto-adjust column widths
                worksheet = writer.sheets["Mediation Results"]
                for idx, col in enumerate(df_results.columns):
                    max_length = (
                        max(df_results[col].astype(str).apply(len).max(), len(str(col)))
                        + 2
                    )
                    worksheet.column_dimensions[get_column_letter(idx + 1)].width = (
                        max_length
                    )

            excel_data = buffer.getvalue()
            st.download_button(
                label="Download Excel",
                data=excel_data,
                file_name="mediation_analysis_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download the results as an Excel file",
            )

    def _render_summary_statistics_tab(self):
        """Render the summary statistics tab."""
        st.header("Summary Statistics")

        if st.session_state.data is None:
            st.warning("Please upload a dataset first.")
            return

        # Display basic statistics
        st.write("### Basic Descriptive Statistics")
        descriptive_stats = st.session_state.data.describe().transpose()
        st.dataframe(descriptive_stats)

        # Options for more detailed statistics
        if st.checkbox("Show detailed statistics", value=False):
            st.write("### Detailed Descriptive Statistics")
            detailed_stats = st.session_state.data.describe(include="all").transpose()
            st.dataframe(detailed_stats)

    def _render_main_content(self):
        """Render the main content with all analysis tabs."""
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
            [
                "Summary Statistics",
                "Normality Analysis",
                "Correlation Analysis",
                "Statistical Tests",
                "Regression Analysis",
                "Mediation Analysis",
                "Sensitivity Analysis",
                "Synthetic Control",
            ],
        )
        with tab1:  # Render the new summary statistics tab
            self._render_summary_statistics_tab()
        with tab2:
            self._render_normality_tab()
        with tab3:
            self._render_correlation_tab()
        with tab4:
            self._render_statistical_tab()
        with tab5:
            self._render_regression_tab()
        with tab6:
            self._render_mediation_tab()
        with tab7:
            self._render_sensitivity_tab()
        with tab8:
            self._render_synthetic_control_tab()

    def _render_sensitivity_tab(self):
        """Render the sensitivity analysis tab."""
        st.header("Sensitivity Analysis")

        if st.session_state.data is None:
            st.warning("Please upload a dataset first.")
            return

        # Get numeric columns
        numeric_cols = self.get_numeric_columns()

        if not numeric_cols:
            st.warning("No numeric columns found in the dataset.")
            return

        # Analysis type selection
        analysis_type = st.radio(
            "Select Analysis Type",
            options=[
                "Missing Data Impact",
                "Outlier Impact",
                "Variable Transformation",
                "Sample Size Impact",
            ],
            help="Choose the type of sensitivity analysis to perform",
        )

        # Variable selection
        target_vars = st.multiselect(
            "Select Variables for Analysis",
            options=numeric_cols,
            help="Select one or more variables to analyze",
        )

        if not target_vars:
            st.warning("Please select at least one variable.")
            return

        # Analysis-specific parameters and execution
        try:
            analyzer = SensitivityAnalyzer()

            if analysis_type == "Missing Data Impact":
                with st.expander("Missing Data Parameters", expanded=True):
                    missing_percentages = st.slider(
                        "Missing Data Percentages",
                        min_value=5,
                        max_value=50,
                        value=(10, 30),
                        step=5,
                        help="Range of missing data percentages to test",
                    )

                    imputation_methods = st.multiselect(
                        "Imputation Methods",
                        options=["Mean", "Median", "Mode", "Linear Interpolation"],
                        default=["Mean", "Median"],
                        help="Select methods for imputing missing values",
                    )

                    n_iterations = st.number_input(
                        "Number of Iterations",
                        min_value=10,
                        max_value=1000,
                        value=100,
                        step=10,
                        help="Number of Monte Carlo iterations",
                    )

                if st.button("Run Analysis"):
                    with st.spinner("Running missing data analysis..."):
                        result = analyzer.analyze_missing_data_impact(
                            st.session_state.data,
                            target_vars,
                            missing_percentages,
                            imputation_methods,
                            n_iterations,
                        )
                        analyzer.create_sensitivity_plots(result)

                        # Download results
                        csv = result.results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results",
                            data=csv,
                            file_name="missing_data_impact_results.csv",
                            mime="text/csv",
                        )

            elif analysis_type == "Outlier Impact":
                with st.expander("Outlier Parameters", expanded=True):
                    outlier_methods = st.multiselect(
                        "Outlier Detection Methods",
                        options=["Z-Score", "IQR"],
                        default=["Z-Score", "IQR"],
                        help="Select methods for detecting outliers",
                    )

                    treatment_methods = st.multiselect(
                        "Outlier Treatment Methods",
                        options=["Remove", "Winsorize", "Cap"],
                        default=["Remove", "Winsorize"],
                        help="Select methods for treating outliers",
                    )

                if st.button("Run Analysis"):
                    with st.spinner("Running outlier analysis..."):
                        result = analyzer.analyze_outlier_impact(
                            st.session_state.data,
                            target_vars,
                            outlier_methods,
                            treatment_methods,
                        )
                        analyzer.create_sensitivity_plots(result)

                        # Download results
                        csv = result.results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results",
                            data=csv,
                            file_name="outlier_impact_results.csv",
                            mime="text/csv",
                        )

            elif analysis_type == "Variable Transformation":
                with st.expander("Transformation Parameters", expanded=True):
                    transformations = st.multiselect(
                        "Transformation Methods",
                        options=["Log", "Square Root", "Box-Cox", "Yeo-Johnson"],
                        default=["Log", "Square Root"],
                        help="Select transformation methods to test",
                    )

                    metrics = st.multiselect(
                        "Evaluation Metrics",
                        options=["Normality", "Skewness", "Kurtosis"],
                        default=["Normality", "Skewness"],
                        help="Select metrics to evaluate transformations",
                    )

                if st.button("Run Analysis"):
                    with st.spinner("Running transformation analysis..."):
                        result = analyzer.analyze_transformation_impact(
                            st.session_state.data, target_vars, transformations, metrics
                        )
                        analyzer.create_sensitivity_plots(result)

                        # Download results
                        csv = result.results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results",
                            data=csv,
                            file_name="transformation_impact_results.csv",
                            mime="text/csv",
                        )

            else:  # Sample Size Impact
                with st.expander("Sample Size Parameters", expanded=True):
                    sample_sizes = st.slider(
                        "Sample Size Percentages",
                        min_value=10,
                        max_value=90,
                        value=(30, 70),
                        step=10,
                        help="Range of sample sizes to test (% of original data)",
                    )

                    n_samples = st.number_input(
                        "Number of Samples per Size",
                        min_value=10,
                        max_value=1000,
                        value=100,
                        step=10,
                        help="Number of random samples to generate for each size",
                    )

                if st.button("Run Analysis"):
                    with st.spinner("Running sample size analysis..."):
                        result = analyzer.analyze_sample_size_impact(
                            st.session_state.data, target_vars, sample_sizes, n_samples
                        )
                        analyzer.create_sensitivity_plots(result)

                        # Download results
                        csv = result.results_df.to_csv(index=False)
                        st.download_button(
                            label="Download Results",
                            data=csv,
                            file_name="sample_size_impact_results.csv",
                            mime="text/csv",
                        )

            # Display additional metrics if available
            if "result" in locals() and result.metrics:
                st.write("### Additional Metrics")
                for key, value in result.metrics.items():
                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")

        except Exception as e:
            st.error(f"Error during sensitivity analysis: {str(e)}")
            st.exception(e)

    def _render_synthetic_control_tab(self):
        """Render the synthetic control analysis tab."""
        st.header("Synthetic Control Analysis")

        if st.session_state.data is None:
            st.warning("Please upload a dataset first.")
            return

        # Get columns by type
        numeric_cols = self.get_numeric_columns()
        categorical_cols = self.get_categorical_columns()

        if not numeric_cols or not categorical_cols:
            st.warning("Dataset must contain both numeric and categorical columns.")
            return

        # Add help text
        st.info(
            """
            Synthetic Control Analysis requires:
            1. A binary treatment variable (0 for control, 1 for treated)
            2. A numeric outcome variable
            3. A time variable
            4. Pre-treatment and post-treatment periods
        """
        )

        # Parameter selection
        col1, col2 = st.columns(2)

        with col1:
            treatment_var = st.selectbox(
                "Treatment Variable (0/1)",
                options=categorical_cols,
                help="Binary variable indicating treatment status",
            )

            outcome_var = st.selectbox(
                "Outcome Variable",
                options=numeric_cols,
                help="Variable to analyze treatment effect",
            )

        with col2:
            time_var = st.selectbox(
                "Time Variable",
                options=numeric_cols,
                help="Variable indicating time periods",
            )

        # Only show treatment time if time variable is selected
        if time_var:
            treatment_time = st.number_input(
                "Treatment Time",
                min_value=float(st.session_state.data[time_var].min()),
                max_value=float(st.session_state.data[time_var].max()),
                value=float(st.session_state.data[time_var].median()),
                help="Time point when treatment occurred",
            )

            # Optional covariates
            covariates = st.multiselect(
                "Additional Covariates (Optional)",
                options=[
                    col for col in numeric_cols if col not in [outcome_var, time_var]
                ],
                help="Additional variables to improve matching",
            )

            if st.button("Run Synthetic Control Analysis"):
                try:
                    # Validate treatment variable
                    if st.session_state.data[treatment_var].nunique() != 2:
                        st.error(
                            f"Treatment variable must be binary (0/1). Found {st.session_state.data[treatment_var].nunique()} unique values."
                        )
                        return

                    # Validate time periods
                    pre_period = st.session_state.data[
                        st.session_state.data[time_var] < treatment_time
                    ]
                    post_period = st.session_state.data[
                        st.session_state.data[time_var] >= treatment_time
                    ]

                    if len(pre_period) == 0:
                        st.error("No pre-treatment periods found.")
                        return
                    if len(post_period) == 0:
                        st.error("No post-treatment periods found.")
                        return

                    with st.spinner("Running synthetic control analysis..."):
                        analyzer = SyntheticControlAnalyzer()
                        result = analyzer.perform_analysis(
                            data=st.session_state.data,
                            treatment_var=treatment_var,
                            outcome_var=outcome_var,
                            time_var=time_var,
                            treatment_time=treatment_time,
                            covariates=covariates if covariates else None,
                        )

                        # Display results
                        st.subheader("Results")

                        # Key metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Treatment Effect", f"{result.treatment_effect:.4f}"
                            )
                        with col2:
                            st.metric("Pre-treatment RMSE", f"{result.pre_rmse:.4f}")
                        with col3:
                            st.metric("R-squared", f"{result.metrics['r_squared']:.4f}")

                        # Plots
                        st.subheader("Visualizations")
                        analyzer.create_plots(result)

                        # Detailed results in expander
                        with st.expander("Detailed Results", expanded=False):
                            st.write("### Unit Weights")
                            weights_df = pd.DataFrame(
                                {
                                    "Control Unit": range(len(result.weights)),
                                    "Weight": result.weights,
                                }
                            )
                            weights_df = weights_df[
                                weights_df["Weight"] > 0.001
                            ]  # Show only significant weights
                            st.dataframe(weights_df)

                            # Download results
                            results_dict = {
                                "time": result.time_values,
                                "treated": result.treated_values,
                                "synthetic": result.synthetic_values,
                                "gap": result.treated_values - result.synthetic_values,
                            }
                            results_df = pd.DataFrame(results_dict)

                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results",
                                data=csv,
                                file_name="synthetic_control_results.csv",
                                mime="text/csv",
                            )

                except Exception as e:
                    st.error(f"Error during synthetic control analysis: {str(e)}")
                    st.exception(e)

    def run(self):
        st.title("DocStat: Statistical Analysis Tool")

        # Sidebar for data upload and basic info
        with st.sidebar:
            self._render_sidebar()

        # Main content
        if st.session_state.data is not None:
            self._render_main_content()
        else:
            st.info("Please upload a dataset to begin analysis.")

    def _render_sidebar(self):
        st.header("Data Upload")
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])

        if uploaded_file is not None:
            df = DataLoader.load_data(uploaded_file)
            if df is not None and DataLoader.validate_data(df):
                st.session_state.data = df
                st.success("Data loaded successfully!")

                # Add categorical column specification
                st.header("Column Settings")
                numeric_cols = df.select_dtypes(
                    include=["int64", "float64"]
                ).columns.tolist()
                if numeric_cols:
                    st.write("Specify numeric columns to treat as categorical:")
                    st.session_state.categorical_numeric_cols = st.multiselect(
                        "Select columns",
                        options=numeric_cols,
                        default=st.session_state.categorical_numeric_cols,
                        help="Select numeric columns that should be treated as categorical variables",
                    )

                # Display data info
                st.header("Dataset Info")
                st.write(f"Rows: {df.shape[0]}")
                st.write(f"Columns: {df.shape[1]}")

                # Display column types
                numeric_cols = self.get_numeric_columns()
                categorical_cols = self.get_categorical_columns()
                st.write("Numeric columns:", len(numeric_cols))
                st.write("Categorical columns:", len(categorical_cols))


if __name__ == "__main__":
    app = DocStatApp()
    app.run()
