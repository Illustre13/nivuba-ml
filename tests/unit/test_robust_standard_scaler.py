# Standard library imports
import os
import sys

# Third-party library imports
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np
from numpy.testing import assert_array_equal
from sklearn.preprocessing import StandardScaler

# Add the root directory to the path for local imports (by going up two levels from current directory in which this file lives)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Local imports
from src.custom_transformers import RobustStandardScaler


# --- Fixtures ---
# Fixture to instantiate FeatureSelector class for use in tests
@pytest.fixture
def transformer():
    return RobustStandardScaler()

# Fixture to create X input DataFrame for use in tests
@pytest.fixture
def X_input():
    return pd.DataFrame({
        "income": [300000, 300000, 300000, 500000, 500000, 500000],
        "age": [30, 30, 30, 50, 50, 50],
        "experience": [3, 3, 3, 5, 5, 5],
        "current_job_yrs": [3, 3, 3, 5, 5, 5],
        "current_house_yrs": [11, 11, 11, 13, 13, 13],
        "state_default_rate": [0.25, 0.25, 0.25, 0.75, 0.75, 0.75],
    })


# --- TestRobustStandardScaler class ---
class TestRobustStandardScaler:
    # Class instantiation 
    @pytest.mark.unit
    def test_instantiation(self, transformer):
        assert isinstance(transformer, StandardScaler)
        assert isinstance(transformer, RobustStandardScaler)

    # Ensure .transform() z-score scales columns
    @pytest.mark.unit
    def test_robust_standard_scaler_happy_path(self, transformer, X_input):
        X = X_input.copy()
        transformer.fit(X)
        X_transformed = transformer.transform(X)
        expected_X_transformed = np.array([
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],                       
        ])
        assert_array_equal(X_transformed, expected_X_transformed)

    # Ensure .transform() correctly handles an empty DataFrame
    @pytest.mark.unit
    def test_transform_handles_empty_df(self, transformer, X_input):
        X = X_input.copy()
        X_empty = pd.DataFrame(columns=X.columns)
        # Fit on non-empty DataFrame, but transform on empty DataFrame
        transformer.fit(X)
        X_transformed_empty = transformer.transform(X_empty)
        # Create expected output (empty numpy array with 0 rows and column equal to input DataFrame)
        n_columns = X.shape[1]
        expected_X_transformed_empty = np.empty((0, n_columns))
        # Ensure output is as expected
        assert_array_equal(X_transformed_empty, expected_X_transformed_empty)
