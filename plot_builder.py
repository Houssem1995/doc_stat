import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


class PlotBuilder:
    """Class to handle all plotting functionality for the application."""

    def __init__(self):
        """Initialize the plot builder."""
        pass

    @staticmethod
    def get_numeric_columns(data):
        """Get numeric columns from the dataframe."""
        return data.select_dtypes(include=["int64", "float64"]).columns.tolist()

    @staticmethod
    def get_categorical_columns(data):
        """Get categorical columns from the dataframe."""
        return data.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

    def render_plotting_tab(self, data):
        """Render the custom plotting tab with various visualization options."""
        st.header("Custom Plot Builder")

        if data is None:
            st.warning("Please upload a dataset first.")
            return

        # Get column types
        numeric_cols = self.get_numeric_columns(data)
        categorical_cols = self.get_categorical_columns(data)
        all_cols = data.columns.tolist()

        # Plot mode selection
        plot_mode = st.radio(
            "Select Plot Mode",
            options=[
                "Two-Variable Plots",
                "Single-Variable Categorical Plots",
                "Single-Variable Numeric Plots",
            ],
            horizontal=True,
        )

        if plot_mode == "Two-Variable Plots":
            self._render_two_variable_plots(
                data, all_cols, numeric_cols, categorical_cols
            )
        elif plot_mode == "Single-Variable Categorical Plots":
            self._render_single_variable_categorical_plots(data, all_cols)
        elif plot_mode == "Single-Variable Numeric Plots":
            self._render_single_variable_numeric_plots(data, all_cols)

    def _render_two_variable_plots(
        self, data, all_cols, numeric_cols, categorical_cols
    ):
        """Render the interface for two-variable plots."""
        # Variable selection section
        st.subheader("1. Select Variables")
        col1, col2 = st.columns(2)

        with col1:
            x_var = st.selectbox(
                "Select X-axis variable",
                all_cols,
                help="Variable for the x-axis of your plot",
                key="plot_x_var",
            )

            # Determine if x is categorical or numeric
            x_is_categorical = x_var in categorical_cols

        with col2:
            y_vars = st.multiselect(
                "Select Y-axis variable(s)",
                [col for col in all_cols if col != x_var],
                help="Select up to 3 variables for the y-axis",
                max_selections=3,
                key="plot_y_vars",
            )

        if not y_vars:
            st.info("Please select at least one y-axis variable to create a plot.")
            return

        # Plot type selection based on variable types
        st.subheader("2. Select Plot Type")

        if x_is_categorical:
            plot_types = [
                "Bar Chart",
                "Pie Chart",
                "Box Plot",
                "Violin Plot",
                "Count Plot",
            ]
        else:
            plot_types = [
                "Line Chart",
                "Scatter Plot",
                "Area Chart",
                "Histogram",
                "Density Plot",
            ]

        # Add common plots that work with either type
        plot_types.extend(["Heatmap", "Bubble Chart"])

        plot_type = st.selectbox(
            "Select plot type", plot_types, key="plot_type_two_var"
        )

        # Plot customization options
        st.subheader("3. Customize Plot")

        with st.expander("Visual Customization", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                # Color options
                use_custom_colors = st.checkbox(
                    "Use custom colors", value=False, key="use_custom_colors_two_var"
                )
                colors = []

                if use_custom_colors:
                    for i, y_var in enumerate(y_vars):
                        default_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
                        colors.append(
                            st.color_picker(
                                f"Color for {y_var}",
                                default_colors[i % len(default_colors)],
                                key=f"color_picker_{i}_two_var",
                            )
                        )

                # Show percentages option (only for certain plot types)
                if plot_type in ["Bar Chart", "Pie Chart", "Count Plot"]:
                    show_percentages = st.checkbox(
                        "Show percentages",
                        value=False,
                        help="Display percentage values on the plot",
                        key="show_percentages_two_var",
                    )
                else:
                    show_percentages = False

            with col2:
                # Title and labels
                plot_title = st.text_input("Plot title", "", key="plot_title_two_var")
                x_label = st.text_input("X-axis label", x_var, key="x_label_two_var")
                y_label = st.text_input(
                    "Y-axis label",
                    ", ".join(y_vars) if len(y_vars) > 1 else y_vars[0],
                    key="y_label_two_var",
                )

                # Figure size
                fig_width = st.slider(
                    "Figure width", 6, 20, 10, key="fig_width_two_var"
                )
                fig_height = st.slider(
                    "Figure height", 4, 15, 6, key="fig_height_two_var"
                )

        # Generate the plot button for two-variable mode
        if st.button("Generate Plot", type="primary", key="generate_plot_two_var"):
            self._generate_two_variable_plot(
                data,
                plot_type,
                x_var,
                y_vars,
                x_is_categorical,
                numeric_cols,
                categorical_cols,
                use_custom_colors,
                colors,
                show_percentages,
                plot_title,
                x_label,
                y_label,
                fig_width,
                fig_height,
            )

    def _generate_two_variable_plot(
        self,
        data,
        plot_type,
        x_var,
        y_vars,
        x_is_categorical,
        numeric_cols,
        categorical_cols,
        use_custom_colors,
        colors,
        show_percentages,
        plot_title,
        x_label,
        y_label,
        fig_width,
        fig_height,
    ):
        """Generate a two-variable plot based on selected parameters."""
        st.subheader("Plot Output")

        try:
            # Handle different plot types for two-variable plots
            if plot_type == "Bar Chart":
                for i, y_var in enumerate(y_vars):
                    color = colors[i] if use_custom_colors and i < len(colors) else None

                    # For categorical x-axis, we might want to aggregate
                    if y_var in numeric_cols:
                        # Aggregate numeric y values by x category
                        plot_data = data.groupby(x_var)[y_var].mean().reset_index()
                        fig = px.bar(
                            plot_data,
                            x=x_var,
                            y=y_var,
                            color_discrete_sequence=[color] if color else None,
                            title=plot_title or f"Average {y_var} by {x_var}",
                        )
                    else:
                        # Count occurrences for categorical y
                        plot_data = (
                            data.groupby([x_var, y_var])
                            .size()
                            .reset_index(name="count")
                        )
                        fig = px.bar(
                            plot_data,
                            x=x_var,
                            y="count",
                            color=y_var,
                            title=plot_title or f"Count of {y_var} by {x_var}",
                        )

                    if show_percentages:
                        # Calculate and display percentages
                        totals = (
                            plot_data[y_var].sum()
                            if y_var in numeric_cols
                            else plot_data["count"].sum()
                        )
                        percentages = [
                            (val / totals * 100)
                            for val in (
                                plot_data[y_var]
                                if y_var in numeric_cols
                                else plot_data["count"]
                            )
                        ]
                        fig.update_traces(
                            text=[f"{p:.1f}%" for p in percentages], textposition="auto"
                        )

                    # Update layout
                    fig.update_layout(
                        xaxis_title=x_label,
                        yaxis_title=y_label or y_var,
                        height=fig_height * 70,
                        width=fig_width * 70,
                    )

                    st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Pie Chart":
                for i, y_var in enumerate(y_vars):
                    # For pie charts, we need to aggregate data
                    if y_var in numeric_cols:
                        # Sum numeric values by category
                        plot_data = data.groupby(x_var)[y_var].sum()
                    else:
                        # Count occurrences for categorical
                        plot_data = (
                            data.groupby([x_var, y_var])
                            .size()
                            .reset_index(name="count")
                        )
                        plot_data = plot_data.groupby(x_var)["count"].sum()

                    # Custom colors handling
                    if use_custom_colors:
                        color = colors[i] if i < len(colors) else None
                        fig = go.Figure(
                            data=[
                                go.Pie(
                                    labels=plot_data.index,
                                    values=plot_data.values,
                                    marker_colors=(
                                        [color] * len(plot_data) if color else None
                                    ),
                                )
                            ]
                        )
                    else:
                        fig = go.Figure(
                            data=[
                                go.Pie(labels=plot_data.index, values=plot_data.values)
                            ]
                        )

                    if show_percentages:
                        fig.update_traces(textinfo="percent+label")
                    else:
                        fig.update_traces(textinfo="label+value")

                    # Update layout
                    fig.update_layout(
                        title=plot_title or f"Distribution of {y_var} by {x_var}",
                        height=fig_height * 70,
                        width=fig_width * 70,
                    )

                    st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Line Chart":
                for i, y_var in enumerate(y_vars):
                    if y_var not in numeric_cols:
                        st.warning(
                            f"{y_var} is not numeric and can't be used in a line chart."
                        )
                        continue

                    color = colors[i] if use_custom_colors and i < len(colors) else None

                    # For line charts, we might want to aggregate by x
                    if x_is_categorical:
                        plot_data = data.groupby(x_var)[y_var].mean().reset_index()
                    else:
                        plot_data = data

                    fig = px.line(
                        plot_data,
                        x=x_var,
                        y=y_var,
                        color_discrete_sequence=[color] if color else None,
                        title=plot_title or f"Trend of {y_var} by {x_var}",
                    )

                    # Update layout
                    fig.update_layout(
                        xaxis_title=x_label,
                        yaxis_title=y_label or y_var,
                        height=fig_height * 70,
                        width=fig_width * 70,
                    )

                    st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Scatter Plot":
                for i, y_var in enumerate(y_vars):
                    if y_var not in numeric_cols:
                        st.warning(
                            f"{y_var} is not numeric and can't be used in a scatter plot."
                        )
                        continue

                    color = colors[i] if use_custom_colors and i < len(colors) else None

                    fig = px.scatter(
                        data,
                        x=x_var,
                        y=y_var,
                        color_discrete_sequence=[color] if color else None,
                        title=plot_title or f"Scatter plot of {y_var} vs {x_var}",
                    )

                    # Update layout
                    fig.update_layout(
                        xaxis_title=x_label,
                        yaxis_title=y_label or y_var,
                        height=fig_height * 70,
                        width=fig_width * 70,
                    )

                    st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Box Plot":
                for i, y_var in enumerate(y_vars):
                    if y_var not in numeric_cols:
                        st.warning(
                            f"{y_var} is not numeric and can't be used in a box plot."
                        )
                        continue

                    color = colors[i] if use_custom_colors and i < len(colors) else None

                    fig = px.box(
                        data,
                        x=x_var,
                        y=y_var,
                        color=x_var if len(y_vars) > 1 else None,
                        color_discrete_sequence=(
                            [color] if color and len(y_vars) == 1 else None
                        ),
                        title=plot_title or f"Box plot of {y_var} by {x_var}",
                    )

                    # Update layout
                    fig.update_layout(
                        xaxis_title=x_label,
                        yaxis_title=y_label or y_var,
                        height=fig_height * 70,
                        width=fig_width * 70,
                    )

                    st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Violin Plot":
                for i, y_var in enumerate(y_vars):
                    if y_var not in numeric_cols:
                        st.warning(
                            f"{y_var} is not numeric and can't be used in a violin plot."
                        )
                        continue

                    color = colors[i] if use_custom_colors and i < len(colors) else None

                    fig = px.violin(
                        data,
                        x=x_var,
                        y=y_var,
                        color=x_var if len(y_vars) > 1 else None,
                        color_discrete_sequence=(
                            [color] if color and len(y_vars) == 1 else None
                        ),
                        box=True,  # Include box plot inside the violin
                        title=plot_title or f"Violin plot of {y_var} by {x_var}",
                    )

                    # Update layout
                    fig.update_layout(
                        xaxis_title=x_label,
                        yaxis_title=y_label or y_var,
                        height=fig_height * 70,
                        width=fig_width * 70,
                    )

                    st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Histogram":
                for i, y_var in enumerate(y_vars):
                    if y_var not in numeric_cols:
                        st.warning(
                            f"{y_var} is not numeric and can't be used in a histogram."
                        )
                        continue

                    color = colors[i] if use_custom_colors and i < len(colors) else None

                    fig = px.histogram(
                        data,
                        x=y_var,
                        color=x_var if x_is_categorical else None,
                        color_discrete_sequence=(
                            [color] if color and not x_is_categorical else None
                        ),
                        title=plot_title or f"Histogram of {y_var}",
                    )

                    # Update layout
                    fig.update_layout(
                        xaxis_title=y_label or y_var,
                        yaxis_title="Count",
                        height=fig_height * 70,
                        width=fig_width * 70,
                    )

                    st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Density Plot":
                for i, y_var in enumerate(y_vars):
                    if y_var not in numeric_cols:
                        st.warning(
                            f"{y_var} is not numeric and can't be used in a density plot."
                        )
                        continue

                    color = colors[i] if use_custom_colors and i < len(colors) else None

                    fig = px.density_contour(
                        data,
                        x=x_var,
                        y=y_var,
                        color_discrete_sequence=[color] if color else None,
                        title=plot_title or f"Density plot of {y_var} vs {x_var}",
                    )

                    # Update layout
                    fig.update_layout(
                        xaxis_title=x_label,
                        yaxis_title=y_label or y_var,
                        height=fig_height * 70,
                        width=fig_width * 70,
                    )

                    st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Area Chart":
                for i, y_var in enumerate(y_vars):
                    if y_var not in numeric_cols:
                        st.warning(
                            f"{y_var} is not numeric and can't be used in an area chart."
                        )
                        continue

                    color = colors[i] if use_custom_colors and i < len(colors) else None

                    # For area charts, we might want to aggregate by x
                    if x_is_categorical:
                        plot_data = data.groupby(x_var)[y_var].mean().reset_index()
                    else:
                        # Need to sort by x for proper area chart
                        plot_data = data.sort_values(by=x_var)

                    fig = px.area(
                        plot_data,
                        x=x_var,
                        y=y_var,
                        color_discrete_sequence=[color] if color else None,
                        title=plot_title or f"Area chart of {y_var} by {x_var}",
                    )

                    # Update layout
                    fig.update_layout(
                        xaxis_title=x_label,
                        yaxis_title=y_label or y_var,
                        height=fig_height * 70,
                        width=fig_width * 70,
                    )

                    st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Heatmap":
                # Heatmap requires a different approach - we need to pivot the data
                if len(y_vars) == 1 and y_vars[0] in numeric_cols:
                    y_var = y_vars[0]

                    # Create a pivot table
                    if x_is_categorical:
                        # Need another categorical variable to create heatmap
                        other_cat_cols = [
                            col for col in categorical_cols if col != x_var
                        ]

                        if not other_cat_cols:
                            st.warning(
                                "Heatmap requires another categorical variable. None available."
                            )
                            return

                        # Let user select the other categorical variable
                        other_cat = st.selectbox(
                            "Select second categorical variable for heatmap:",
                            options=other_cat_cols,
                        )

                        # Create pivot table
                        pivot_data = pd.pivot_table(
                            data,
                            values=y_var,
                            index=x_var,
                            columns=other_cat,
                            aggfunc="mean",
                        )

                        fig = px.imshow(
                            pivot_data,
                            title=plot_title
                            or f"Heatmap of {y_var} by {x_var} and {other_cat}",
                            color_continuous_scale=(
                                "RdBu_r" if not use_custom_colors else None
                            ),
                        )

                        # Update layout
                        fig.update_layout(
                            xaxis_title=other_cat,
                            yaxis_title=x_var,
                            height=fig_height * 70,
                            width=fig_width * 70,
                        )

                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(
                            "Heatmap requires categorical variables. Try selecting a categorical X variable."
                        )
                else:
                    st.warning("Heatmap requires exactly one numeric Y variable.")

            elif plot_type == "Bubble Chart":
                if len(y_vars) >= 1 and all(var in numeric_cols for var in y_vars):
                    # For bubble chart, we need at least one y variable, and optionally a size variable
                    y_var = y_vars[0]
                    size_var = y_vars[1] if len(y_vars) > 1 else None

                    if size_var:
                        fig = px.scatter(
                            data,
                            x=x_var,
                            y=y_var,
                            size=size_var,
                            color=y_vars[2] if len(y_vars) > 2 else None,
                            color_discrete_sequence=(
                                colors if use_custom_colors else None
                            ),
                            title=plot_title
                            or f"Bubble chart of {y_var} vs {x_var} (size: {size_var})",
                        )
                    else:
                        # No size variable provided, use constant size
                        fig = px.scatter(
                            data,
                            x=x_var,
                            y=y_var,
                            size_max=20,
                            color_discrete_sequence=(
                                colors if use_custom_colors else None
                            ),
                            title=plot_title or f"Bubble chart of {y_var} vs {x_var}",
                        )

                    # Update layout
                    fig.update_layout(
                        xaxis_title=x_label,
                        yaxis_title=y_label or y_var,
                        height=fig_height * 70,
                        width=fig_width * 70,
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(
                        "Bubble chart requires at least one numeric Y variable. Optional second numeric Y variable for bubble size."
                    )

            elif plot_type == "Count Plot":
                # Count plot is for counting occurrences of categorical data
                if x_is_categorical:
                    counts = data[x_var].value_counts().reset_index()
                    counts.columns = [x_var, "count"]

                    fig = px.bar(
                        counts,
                        x=x_var,
                        y="count",
                        color=(
                            x_var
                            if len(y_vars) > 0 and y_vars[0] in categorical_cols
                            else None
                        ),
                        color_discrete_sequence=colors if use_custom_colors else None,
                        title=plot_title or f"Count of {x_var}",
                    )

                    if show_percentages:
                        total = counts["count"].sum()
                        percentages = [
                            (count / total * 100) for count in counts["count"]
                        ]
                        fig.update_traces(
                            text=[f"{p:.1f}%" for p in percentages], textposition="auto"
                        )

                    # Update layout
                    fig.update_layout(
                        xaxis_title=x_label,
                        yaxis_title="Count",
                        height=fig_height * 70,
                        width=fig_width * 70,
                    )

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Count plot requires a categorical X variable.")

            # Add download options
            st.subheader("4. Export Plot")
            st.caption(
                "You can download the plot by clicking the camera icon in the top-right corner of each plot."
            )

        except Exception as e:
            st.error(f"Error generating plot: {str(e)}")
            st.exception(e)

    def _render_single_variable_categorical_plots(self, data, categorical_cols):
        """Render the interface for single-variable categorical plots."""
        st.subheader("Select a Categorical Variable")
        categorical_var = st.selectbox(
            "Choose Variable", categorical_cols, key="categorical_var"
        )

        # Add mapping functionality
        st.subheader("Category Mapping")
        use_mapping = st.checkbox(
            "Use category mapping", value=False, key="use_mapping"
        )

        mapping_dict = {}
        if use_mapping:
            unique_values = sorted(data[categorical_var].unique())
            st.write("Define mappings for each category:")

            # Create two columns for better layout
            cols = st.columns(2)
            for idx, val in enumerate(unique_values):
                # Alternate between columns
                with cols[idx % 2]:
                    mapped_value = st.text_input(
                        f"Label for {val}",
                        value=str(val),
                        key=f"mapping_{categorical_var}_{val}",
                    )
                    mapping_dict[val] = mapped_value

        plot_type = st.selectbox(
            "Select Plot Type",
            ["Bar Chart", "Pie Chart", "Donut Chart"],
            key="categorical_plot_type",
        )

        # Plot customization options
        with st.expander("Visual Customization", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                # Color options
                use_custom_colors = st.checkbox(
                    "Use custom colors", value=False, key="use_custom_colors_cat"
                )

                if use_custom_colors:
                    unique_values = sorted(data[categorical_var].unique())
                    category_colors = {}
                    for val in unique_values:
                        display_val = mapping_dict.get(val, val) if use_mapping else val
                        category_colors[val] = st.color_picker(
                            f"Color for {display_val}",
                            key=f"color_{categorical_var}_{val}",
                        )

                show_percentages = st.checkbox(
                    "Show percentages",
                    value=False,
                    help="Display percentage values on the plot",
                    key="show_percentages_cat",
                )

            with col2:
                plot_title = st.text_input("Plot title", "", key="plot_title_cat")
                fig_width = st.slider("Figure width", 6, 20, 10, key="fig_width_cat")
                fig_height = st.slider("Figure height", 4, 15, 6, key="fig_height_cat")

        if st.button("Generate Plot", type="primary", key="generate_plot_cat"):
            # Apply mapping to data if enabled
            plot_data = data.copy()
            if use_mapping:
                plot_data[categorical_var] = plot_data[categorical_var].map(
                    mapping_dict
                )

            self._generate_single_variable_categorical_plot(
                plot_data,
                categorical_var,
                plot_type,
                use_custom_colors,
                category_colors if use_custom_colors else None,
                show_percentages,
                plot_title,
                fig_width,
                fig_height,
            )

    def _generate_single_variable_categorical_plot(
        self,
        data,
        categorical_var,
        cat_plot_type,
        use_custom_colors,
        category_colors,
        show_percentages,
        plot_title,
        fig_width,
        fig_height,
    ):
        """Generate a single-variable categorical plot based on selected parameters."""
        st.subheader("Plot Output")

        try:
            # Get category counts
            value_counts = data[categorical_var].value_counts()

            # Create different plot types for categorical variables
            if cat_plot_type == "Count Plot":
                # Create a bar chart of counts
                if use_custom_colors:
                    # Map colors to categories
                    color_map = {
                        cat: category_colors.get(cat, "#1f77b4")
                        for cat in value_counts.index
                    }
                    colors = [color_map[cat] for cat in value_counts.index]
                    fig = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        color=value_counts.index,
                        color_discrete_map=color_map,
                        title=plot_title,
                    )
                else:
                    fig = px.bar(
                        x=value_counts.index, y=value_counts.values, title=plot_title
                    )

                # Add percentages if requested
                if show_percentages:
                    total = value_counts.sum()
                    percentages = [
                        (count / total * 100) for count in value_counts.values
                    ]
                    fig.update_traces(
                        text=[f"{p:.1f}%" for p in percentages], textposition="auto"
                    )

                # Update layout
                fig.update_layout(
                    xaxis_title=categorical_var,
                    yaxis_title="Count",
                    height=fig_height * 70,
                    width=fig_width * 70,
                )

            elif cat_plot_type == "Pie Chart" or cat_plot_type == "Donut Chart":
                # Create a pie chart
                if use_custom_colors:
                    # Map colors to categories
                    color_map = {
                        cat: category_colors.get(cat, "#1f77b4")
                        for cat in value_counts.index
                    }
                    colors = [color_map[cat] for cat in value_counts.index]
                    fig = px.pie(
                        names=value_counts.index,
                        values=value_counts.values,
                        color=value_counts.index,
                        color_discrete_map=color_map,
                        title=plot_title,
                    )
                else:
                    fig = px.pie(
                        names=value_counts.index,
                        values=value_counts.values,
                        title=plot_title,
                    )

                # Configure text based on percentage display option
                if show_percentages:
                    fig.update_traces(textinfo="percent+label")
                else:
                    fig.update_traces(textinfo="label+value")

                # Create a donut chart if selected
                if cat_plot_type == "Donut Chart":
                    fig.update_traces(hole=0.4)

                # Update layout
                fig.update_layout(height=fig_height * 70, width=fig_width * 70)

            elif cat_plot_type == "Bar Chart":
                # Create a horizontal bar chart
                if use_custom_colors:
                    # Map colors to categories
                    color_map = {
                        cat: category_colors.get(cat, "#1f77b4")
                        for cat in value_counts.index
                    }
                    colors = [color_map[cat] for cat in value_counts.index]
                    fig = px.bar(
                        y=value_counts.index,
                        x=value_counts.values,
                        color=value_counts.index,
                        color_discrete_map=color_map,
                        orientation="h",
                        title=plot_title,
                    )
                else:
                    fig = px.bar(
                        y=value_counts.index,
                        x=value_counts.values,
                        orientation="h",
                        title=plot_title,
                    )

                # Add percentages if requested
                if show_percentages:
                    total = value_counts.sum()
                    percentages = [
                        (count / total * 100) for count in value_counts.values
                    ]
                    fig.update_traces(
                        text=[f"{p:.1f}%" for p in percentages], textposition="auto"
                    )

                # Update layout
                fig.update_layout(
                    yaxis_title=categorical_var,
                    xaxis_title="Count",
                    height=fig_height * 70,
                    width=fig_width * 70,
                )

            # Display the plot
            st.plotly_chart(fig, use_container_width=True)

            # Display summary statistics
            with st.expander("Category Summary", expanded=False):
                # Calculate summary statistics
                total = value_counts.sum()
                percentages = [(count / total * 100) for count in value_counts.values]

                # Create summary dataframe
                summary_df = pd.DataFrame(
                    {
                        "Category": value_counts.index,
                        "Count": value_counts.values,
                        "Percentage": [f"{p:.2f}%" for p in percentages],
                    }
                )

                st.dataframe(summary_df)

                # Download option for summary
                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="Download Summary",
                    data=csv,
                    file_name=f"{categorical_var}_distribution.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error generating plot: {str(e)}")
            st.exception(e)

    def _render_single_variable_numeric_plots(self, data, numeric_cols):
        """Render the interface for single-variable numeric plots."""
        st.subheader("Select a Numeric Variable")
        numeric_var = st.selectbox("Choose Variable", numeric_cols, key="numeric_var")

        plot_type = st.selectbox(
            "Select Plot Type",
            ["Histogram", "Box Plot", "Violin Plot", "Density Plot"],
            key="numeric_plot_type",
        )

        # Plot customization options
        with st.expander("Customize Plot", expanded=True):
            bin_size = (
                st.slider(
                    "Bin Size", min_value=1, max_value=100, value=20, key="bin_size"
                )
                if plot_type == "Histogram"
                else None
            )
            use_custom_color = st.checkbox(
                "Use Custom Color", value=False, key="use_custom_color_numeric"
            )
            custom_color = (
                st.color_picker("Pick a Color", key="custom_color_numeric")
                if use_custom_color
                else None
            )

        if st.button("Generate Plot", key="generate_numeric_plot"):
            self._generate_single_variable_numeric_plot(
                data, numeric_var, plot_type, bin_size, custom_color
            )

    def _generate_single_variable_numeric_plot(
        self, data, numeric_var, plot_type, bin_size, custom_color
    ):
        """Generate the selected type of plot for a single numeric variable."""
        if plot_type == "Histogram":
            fig = px.histogram(
                data,
                x=numeric_var,
                nbins=bin_size,
                title=f"Histogram of {numeric_var}",
                color_discrete_sequence=[custom_color] if custom_color else None,
            )
        elif plot_type == "Box Plot":
            fig = px.box(
                data,
                y=numeric_var,
                title=f"Box Plot of {numeric_var}",
                color_discrete_sequence=[custom_color] if custom_color else None,
            )
        elif plot_type == "Violin Plot":
            fig = px.violin(
                data,
                y=numeric_var,
                box=True,
                points="all",
                title=f"Violin Plot of {numeric_var}",
                color_discrete_sequence=[custom_color] if custom_color else None,
            )
        elif plot_type == "Density Plot":
            fig = px.density_contour(
                data,
                x=numeric_var,
                y=numeric_var,
                title=f"Density Plot of {numeric_var}",
                color_discrete_sequence=[custom_color] if custom_color else None,
            )
        else:
            st.error("Unsupported plot type selected.")
            return

        st.plotly_chart(fig, use_container_width=True)
