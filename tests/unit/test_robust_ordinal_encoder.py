# Standard library imports
import os
import sys

# Third-party library imports
import pytest
import pandas as pd
import numpy as np
from numpy.testing import assert_array_equal
from sklearn.preprocessing import OrdinalEncoder

# Add the root directory to the path for local imports (by going up two levels from current directory in which this file lives)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Local imports
from src.custom_transformers import RobustOrdinalEncoder
from src.global_constants import ORDINAL_COLUMN_ORDERS


# --- Fixtures ---
# Fixture to instantiate FeatureSelector class for use in tests
@pytest.fixture
def transformer():
    return RobustOrdinalEncoder(categories=ORDINAL_COLUMN_ORDERS)

# Fixture to create X input DataFrame for use in tests
@pytest.fixture
def X_input():
    return pd.DataFrame({
        "job_stability": ["variable", "moderate", "stable", "very_stable"],
        "city_tier": ["unknown", "tier_3", "tier_2", "tier_1"],
    })


# --- TestRobustStandardScaler class ---
class TestRobustStandardScaler:
    # Class instantiation 
    @pytest.mark.unit
    def test_instantiation(self, transformer):
        assert isinstance(transformer, OrdinalEncoder)
        assert isinstance(transformer, RobustOrdinalEncoder)
        assert transformer.categories == ORDINAL_COLUMN_ORDERS

    # Ensure .transform() encodes ordinal columns "job_stability" and "city_tier"
    @pytest.mark.unit
    def test_robust_ordinal_encoder_happy_path(self, transformer, X_input):
        X = X_input.copy()
        transformer.fit(X)
        X_transformed = transformer.transform(X)
        expected_X_transformed = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],                     
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
