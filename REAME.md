

# DocStat: Statistical Analysis Tool 📊

[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

DocStat is a powerful, user-friendly statistical analysis tool built with Streamlit. It provides an intuitive interface for performing various statistical analyses, including normality tests, correlation analysis, and statistical hypothesis testing.

## 🌟 Features

### Data Analysis
- **File Upload**: Support for CSV and Excel files
- **Automatic Type Detection**: Identifies numeric and categorical variables
- **Data Validation**: Checks for data integrity and completeness

### Statistical Tests
- **Normality Testing**
  - Shapiro-Wilk test
  - D'Agostino-Pearson test
  - Visual analysis (Q-Q plots, histograms)

- **Correlation Analysis**
  - Pearson correlation
  - Spearman correlation
  - Correlation matrices and heatmaps
  - P-value analysis

- **Statistical Tests**
  - Two-group comparisons (T-test/Mann-Whitney)
  - Multiple group comparisons (ANOVA/Kruskal-Wallis)
  - Categorical analysis (Chi-square)

### Visualization
- Interactive plots using Plotly
- Customizable visualizations
- Export options for graphs and results

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- Make (for using Makefile commands)
- Docker (optional, for containerized deployment)


docstat/
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py        # Data loading and preprocessing functions
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── normality.py         # Normality test functions
│   │   ├── correlation.py       # Correlation analysis functions
│   │   └── statistical.py       # Statistical test functions
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── normality_plots.py   # Normality visualization functions
│   │   ├── correlation_plots.py # Correlation visualization functions
│   │   └── statistical_plots.py # Statistical test visualization functions
│   │
│   └── utils/
│       ├── __init__.py
│       └── helpers.py           # Helper functions and utilities
│
├── config/
│   └── settings.py              # Configuration settings
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_analysis.py
│   └── test_visualization.py
│
├── assets/
│   └── style.css               # Custom styling
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
