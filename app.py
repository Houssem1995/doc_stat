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
from src.analysis.mediation import MediationAnalyzer
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from scipy import stats


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
                include=['int64', 'float64']
            ).columns.tolist()
            # Exclude columns marked as categorical
            return [col for col in numeric_cols if col not in st.session_state.categorical_numeric_cols]
        return []

    def get_categorical_columns(self) -> list:
        """Get list of categorical columns from the loaded dataset."""
        if st.session_state.data is not None:
            # Include both native categorical and user-specified categorical columns
            native_categorical = st.session_state.data.select_dtypes(
                include=['object', 'category']
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
            key="correlation_vars"
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
                key="correlation_method"
            )
        
        with col2:
            significance_level = st.slider(
                "Significance Level (α)",
                min_value=0.01,
                max_value=0.10,
                value=0.05,
                step=0.01,
                key="correlation_significance",
                help="Threshold for statistical significance"
            )
        
        if st.button("Perform Correlation Analysis", key="run_correlation") and len(selected_columns) >= 2:
            try:
                # Perform correlation analysis
                results = CorrelationAnalyzer.calculate_correlation(
                    data=st.session_state.data,
                    variables=selected_columns,
                    method=method,
                    significance_level=significance_level
                )
                
                # Display correlation matrix
                st.write("### Correlation Matrix")
                correlation_df = results.correlation_matrix.style.background_gradient(
                    cmap='RdYlBu',
                    vmin=-1,
                    vmax=1
                ).format("{:.3f}")
                st.dataframe(correlation_df)
                
                # Display p-values
                st.write("### P-Values Matrix")
                pvalue_df = results.p_value_matrix.style.background_gradient(
                    cmap='Reds_r'
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
                        columns=['Variable 1', 'Variable 2', 'Correlation']
                    ).sort_values('Correlation', key=abs, ascending=False)
                    
                    # Add significance indicators
                    significant_df['Strength'] = significant_df['Correlation'].apply(
                        lambda x: '🔴 Strong' if abs(x) > 0.7 else
                        '🟡 Moderate' if abs(x) > 0.3 else
                        '🟢 Weak'
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
                        key="download_correlation_matrix"
                    )
                with col2:
                    csv_pval = results.p_value_matrix.to_csv(index=True)
                    st.download_button(
                        label="Download P-Values Matrix",
                        data=csv_pval,
                        file_name="p_values_matrix.csv",
                        mime="text/csv",
                        key="download_pvalues_matrix"
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
            st.markdown("""
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
            """)
        
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
                key="normality_column_select"
            )
        
        with col2:
            # Test selection
            test_method = st.radio(
                "Select Normality Test",
                options=["Shapiro-Wilk", "D'Agostino-Pearson", "Both"],
                help="Choose the normality test method",
                key="normality_test_method"
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
                help="Probability threshold for determining statistical significance"
            )
            
            show_detailed_stats = st.checkbox(
                "Show Detailed Statistics",
                value=False,
                key="show_detailed_stats",
                help="Display additional statistical measures"
            )
        
        if st.button("Perform Normality Tests", key="run_normality") and selected_columns:
            try:
                # Perform analysis
                results = NormalityAnalyzer.analyze_multiple(
                    data=st.session_state.data,
                    columns=selected_columns,
                    significance_level=significance_level,
                    test_method=test_method
                )
                
                # Display summary table
                st.write("### Summary of Normality Tests")
                summary_df = NormalityAnalyzer.create_summary_table(results, test_method)
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
                        key="download_normality_summary"
                    )
                
                if show_detailed_stats:
                    with col2:
                        csv_detailed = detailed_df.to_csv(index=False)
                        st.download_button(
                            label="Download Detailed Statistics",
                            data=csv_detailed,
                            file_name="normality_test_detailed.csv",
                            mime="text/csv",
                            key="download_normality_detailed"
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
            st.warning("Dataset must contain both numeric and categorical columns for statistical tests.")
            return
        
        # Test type selection
        test_type = st.radio(
            "Select Test Type",
            options=[
                "Two Groups (T-test/Mann-Whitney)",
                "Multiple Groups (ANOVA/Kruskal)",
                "Categorical (Chi-square)"
            ],
            help="Select the appropriate test based on your data",
            key="statistical_test_type"
        )
        
        # Variable selection based on test type
        col1, col2 = st.columns(2)
        with col1:
            if "Chi-square" in test_type:
                var1 = st.selectbox(
                    "Select First Variable", 
                    categorical_cols,
                    key="stat_var1"
                )
            else:
                var1 = st.selectbox(
                    "Select Dependent Variable", 
                    numeric_cols,
                    key="stat_var1"
                )
        
        with col2:
            var2 = st.selectbox(
                "Select Independent/Grouping Variable", 
                categorical_cols,
                key="stat_var2"
            )
        
        if st.button("Perform Statistical Test", key="run_statistical"):
            try:
                if "Two Groups" in test_type:
                    result = StatisticalAnalyzer.perform_two_group_test(
                        st.session_state.data, var1, var2
                    )
                elif "Multiple Groups" in test_type:
                    result = StatisticalAnalyzer.perform_multi_group_test(
                        st.session_state.data, var1, var2
                    )
                else:  # Chi-square
                    result = StatisticalAnalyzer.perform_chi_square_test(
                        st.session_state.data, var1, var2
                    )
                
                # Display results
                st.write(f"### {result.test_name} Results")
                st.write(f"Statistic: {result.statistic:.4f}")
                st.write(f"P-value: {result.p_value:.4f}")
                st.write(f"Significant: {'Yes' if result.significant else 'No'}")
                
                # Display plots
                st.write("### Visualization")
                create_statistical_plots(result, test_type)
                
                # Additional information
                if result.additional_info:
                    st.write("### Additional Information")
                    for key, value in result.additional_info.items():
                        st.write(f"{key}: {value}")
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
                key="reg_dependent"
            )
        
        with col2:
            independent_vars = st.multiselect(
                "Select Independent Variable(s) (X)",
                options=[col for col in numeric_cols if col != dependent_var],
                key="reg_independent"
            )
        
        # Analysis options
        with st.expander("Regression Options", expanded=False):
            test_size = st.slider(
                "Test Set Size",
                min_value=0.1,
                max_value=0.4,
                value=0.2,
                step=0.05,
                help="Proportion of data used for testing"
            )
            
            standardize = st.checkbox(
                "Standardize Variables",
                value=True,
                help="Convert variables to z-scores"
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
                    standardize=standardize
                )
                
                # Display results
                st.write("### Model Summary")
                st.write(f"R-squared: {results.r_squared:.3f}")
                st.write(f"Adjusted R-squared: {results.adjusted_r_squared:.3f}")
                st.write(f"RMSE: {results.rmse:.3f}")
                
                # Coefficients table
                st.write("### Coefficients")
                coef_df = pd.DataFrame({
                    'Coefficient': results.coefficients,
                    'Std Error': results.std_errors,
                    'P-value': results.p_values
                }).round(4)
                st.dataframe(coef_df)
                
                # Plots
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### Residuals Plot")
                    fig_residuals = px.scatter(
                        x=results.predictions,
                        y=results.residuals,
                        labels={'x': 'Predicted Values', 'y': 'Residuals'}
                    )
                    st.plotly_chart(fig_residuals)
                
                with col2:
                    st.write("### Q-Q Plot")
                    fig_qq = px.scatter(
                        x=np.sort(stats.norm.ppf(np.linspace(0.01, 0.99, len(results.residuals)))),
                        y=np.sort(results.residuals),
                        labels={'x': 'Theoretical Quantiles', 'y': 'Sample Quantiles'}
                    )
                    fig_qq.add_trace(go.Scatter(
                        x=[-3, 3],
                        y=[-3, 3],
                        mode='lines',
                        line=dict(dash='dash'),
                        showlegend=False
                    ))
                    st.plotly_chart(fig_qq)
                
            except Exception as e:
                st.error(f"Error during regression analysis: {str(e)}")
                st.exception(e)

    def _render_mediation_tab(self):
        st.header("Mediation Analysis")
        
        if st.session_state.data is None:
            st.warning("Please upload a dataset first.")
            return
        
        # Get numeric columns
        numeric_cols = self.get_numeric_columns()
        
        if len(numeric_cols) < 3:
            st.warning("Mediation analysis requires at least 3 numeric variables.")
            return
        
        # Variable selection
        col1, col2, col3 = st.columns(3)
        
        with col1:
            independent_var = st.selectbox(
                "Select Independent Variable (X)",
                options=numeric_cols,
                key="med_independent"
            )
        
        with col2:
            mediator = st.selectbox(
                "Select Mediator (M)",
                options=[col for col in numeric_cols if col != independent_var],
                key="med_mediator"
            )
        
        with col3:
            dependent_var = st.selectbox(
                "Select Dependent Variable (Y)",
                options=[col for col in numeric_cols if col not in [independent_var, mediator]],
                key="med_dependent"
            )
        
        # Analysis options
        with st.expander("Mediation Options", expanded=False):
            confidence_level = st.slider(
                "Confidence Level",
                min_value=0.80,
                max_value=0.99,
                value=0.95,
                step=0.01,
                help="Confidence level for bootstrap intervals"
            )
            
            n_bootstrap = st.number_input(
                "Number of Bootstrap Samples",
                min_value=1000,
                max_value=10000,
                value=5000,
                step=1000,
                help="Number of bootstrap samples for confidence intervals"
            )
        
        if st.button("Run Mediation Analysis", key="run_mediation"):
            try:
                results = MediationAnalyzer.perform_mediation(
                    data=st.session_state.data,
                    independent_var=independent_var,
                    mediator=mediator,
                    dependent_var=dependent_var,
                    confidence_level=confidence_level,
                    n_bootstrap=n_bootstrap
                )
                
                # Display results
                st.write("### Mediation Analysis Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Effect Sizes**")
                    st.write(f"Total Effect (c): {results.total_effect:.3f}")
                    st.write(f"Direct Effect (c'): {results.direct_effect:.3f}")
                    st.write(f"Indirect Effect (ab): {results.indirect_effect:.3f}")
                
                with col2:
                    st.write("**Statistical Tests**")
                    st.write(f"Total Effect p-value: {results.total_effect_p:.3f}")
                    st.write(f"Direct Effect p-value: {results.direct_effect_p:.3f}")
                    st.write(f"Sobel Test p-value: {results.sobel_p:.3f}")
                
                st.write("### Bootstrap Results")
                st.write(f"Indirect Effect 95% CI: [{results.indirect_effect_ci[0]:.3f}, {results.indirect_effect_ci[1]:.3f}]")
                st.write(f"Proportion Mediated: {results.proportion_mediated:.1%}")
                
                # Create path diagram
                st.write("### Path Diagram")
                # Add visualization of the mediation model here
                
            except Exception as e:
                st.error(f"Error during mediation analysis: {str(e)}")
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
        uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            df = DataLoader.load_data(uploaded_file)
            if df is not None and DataLoader.validate_data(df):
                st.session_state.data = df
                st.success("Data loaded successfully!")
                
                # Add categorical column specification
                st.header("Column Settings")
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                if numeric_cols:
                    st.write("Specify numeric columns to treat as categorical:")
                    st.session_state.categorical_numeric_cols = st.multiselect(
                        "Select columns",
                        options=numeric_cols,
                        default=st.session_state.categorical_numeric_cols,
                        help="Select numeric columns that should be treated as categorical variables"
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

    def _render_main_content(self):
        """Render the main content with all analysis tabs."""
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Normality Analysis",
            "Correlation Analysis",
            "Statistical Tests",
            "Regression Analysis",
            "Mediation Analysis"
        ])
        
        # Render content for each tab
        with tab1:
            self._render_normality_tab()
        
        with tab2:
            self._render_correlation_tab()
        
        with tab3:
            self._render_statistical_tab()
        
        with tab4:
            self._render_regression_tab()
        
        with tab5:
            self._render_mediation_tab()


if __name__ == "__main__":
    app = DocStatApp()
    app.run()
