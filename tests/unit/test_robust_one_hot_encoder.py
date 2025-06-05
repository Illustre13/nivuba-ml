# Standard library imports
import os
import sys

# Third-party library imports
import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal
from sklearn.preprocessing import OneHotEncoder

# Add the root directory to the path for local imports (by going up two levels from current directory in which this file lives)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Local imports
from src.custom_transformers import RobustOneHotEncoder
from src.global_constants import NOMINAL_COLUMN_CATEGORIES


# --- Fixtures ---
# Fixture to instantiate FeatureSelector class for use in tests
@pytest.fixture
def transformer():
    return RobustOneHotEncoder(categories=NOMINAL_COLUMN_CATEGORIES, drop="first", sparse_output=False)

# Fixture to create X input DataFrame for use in tests
@pytest.fixture
def X_input():
    return pd.DataFrame({
        "house_ownership": ["rented", "rented", "norent_noown", "rented", "rented", "owned"],
    })


# --- TestRobustStandardScaler class ---
class TestRobustStandardScaler:
    # Class instantiation 
    @pytest.mark.unit
    def test_instantiation(self, transformer):
        assert isinstance(transformer, OneHotEncoder)
        assert isinstance(transformer, RobustOneHotEncoder)
        assert transformer.categories == NOMINAL_COLUMN_CATEGORIES

    # Ensure .transform() one-hot encodes "house_ownership" column
    @pytest.mark.unit
    def test_robust_one_hot_encoder_happy_path(self, transformer, X_input):
        X = X_input.copy()
        transformer.fit(X)
        X_transformed = transformer.transform(X)
        expected_X_transformed = np.array([
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],                      
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
        # Create expected output (empty numpy array with 0 rows and 2 columns)
        expected_X_transformed_empty = np.empty((0, 2))
        # Ensure output is as expected
        assert_array_equal(X_transformed_empty, expected_X_transformed_empty)
