# Standard library imports
import os
import sys

# Third-party library imports
import pytest
import pandas as pd
import numpy as np
from pandas.testing import assert_frame_equal
from sklearn.impute import SimpleImputer

# Add the root directory to the path for local imports (by going up two levels from current directory in which this file lives)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Local imports
from src.custom_transformers import RobustSimpleImputer


# --- Fixtures ---
# Fixture to instantiate FeatureSelector class for use in tests
@pytest.fixture
def transformer():
    return RobustSimpleImputer(strategy="most_frequent").set_output(transform="pandas")

# Fixture to create X input DataFrame for use in tests
@pytest.fixture
def X_input():
    return pd.DataFrame({
        "married": [np.nan, "single", "single", "married", "single", "single"],
        "house_ownership": [np.nan, "rented", "norent_noown", "rented", "rented", "owned"],
        "car_ownership": [np.nan, "no", "yes", "no", "no", "no"],
    })


# --- TestRobustSimpleImputer class ---
class TestRobustSimpleImputer:
    # Class instantiation 
    @pytest.mark.unit
    def test_instantiation(self, transformer):
        assert isinstance(transformer, SimpleImputer)
        assert isinstance(transformer, RobustSimpleImputer)

    # Ensure .transform() imputes the mode
    @pytest.mark.unit
    def test_robust_simple_imputer_transform_imputes_mode(self, transformer, X_input):
        X = X_input.copy()
        transformer.fit(X)
        X_transformed = transformer.transform(X)
        expected_X_transformed = pd.DataFrame({
            "married": ["single", "single", "single", "married", "single", "single"],
            "house_ownership": ["rented", "rented", "norent_noown", "rented", "rented", "owned"],
            "car_ownership": ["no", "no", "yes", "no", "no", "no"],
        })
        assert_frame_equal(X_transformed, expected_X_transformed)

    # Ensure .transform() correctly handles an empty DataFrame
    @pytest.mark.unit
    def test_transform_handles_empty_df(self, transformer, X_input):
        X = X_input.copy()
        # Create empty DataFrame
        input_columns = X.columns
        input_data_types = X.dtypes.to_dict()
        X_empty = pd.DataFrame(columns=input_columns).astype(input_data_types)
        # Fit and transform on "full" DataFrame
        transformer.fit(X)
        X_transformed = transformer.transform(X)
        # Transform on empty Dataframe     
        X_transformed_empty = transformer.transform(X_empty)
        # Create expected output (based on "full" DataFrame)
        expected_output_columns = X_transformed.columns
        expected_output_data_types = X_transformed.dtypes.to_dict()
        expected_X_transformed_empty = pd.DataFrame(columns=expected_output_columns).astype(expected_output_data_types)
        # Ensure actual and expected output DataFrames are identical
        assert_frame_equal(X_transformed_empty, expected_X_transformed_empty)
