"""Configuration settings for the DocStat application."""

# Analysis settings
SIGNIFICANCE_LEVEL = 0.05
SAMPLE_SIZE_THRESHOLD = 50  # For choosing between different normality tests

# Visualization settings
PLOT_HEIGHT = 500
PLOT_WIDTH = 800
COLOR_SCHEME = "RdBu"
CATEGORICAL_PALETTE = "Set3"

# File settings
ALLOWED_EXTENSIONS = ['.csv', '.xlsx', '.xls']
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB

# Test configurations
TEST_TYPES = {
    'normality': ['Shapiro-Wilk', "D'Agostino-Pearson"],
    'correlation': ['Pearson', 'Spearman'],
    'statistical': [
        'Two Groups (T-test/Mann-Whitney)',
        'Multiple Groups (ANOVA/Kruskal)',
        'Categorical (Chi-square)'
    ]
} 