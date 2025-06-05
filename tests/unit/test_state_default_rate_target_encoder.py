# Standard library imports
import os
import sys
import re

# Third-party library imports
import pytest
import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal
import numpy as np

# Add the root directory to the path for local imports (by going up two levels from current directory in which this file lives)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Local imports
from src.custom_transformers import StateDefaultRateTargetEncoder, MissingValueError, CategoricalLabelError
from tests.unit.base_transformer_tests import BaseSupervisedTransformerTests


# --- Fixtures ---
# Fixture to instantiate JobStabilityTransformer class for use in tests
@pytest.fixture
def transformer():
    return StateDefaultRateTargetEncoder()

# Fixture to create X input DataFrame for use in tests
@pytest.fixture
def X_input():
    return pd.DataFrame({
        "income": [9121364, 2636544, 9470213, 6558967, 6245331, 154867],
        "age": [70, 39, 41, 41, 65, 64],
        "experience": [18, 0, 5, 10, 6, 1],
        "married": [False, False, False, True, False, False],
        "house_ownership": ["rented", "rented", "norent_noown", "rented", "rented", "owned"],
        "car_ownership": [False, False, True, False, False, False],
        "profession": ["artist", "computer_hardware_engineer", "web_designer", "comedian", 
                       "financial_analyst", "statistician"],
        "city": ["sikar", "vellore", "bidar", "bongaigaon", "eluru[25]", "danapur"],
        "state": ["rajasthan", "tamil_nadu", "karnataka", "assam", "andhra_pradesh", "bihar"],
        "current_job_yrs": [3, 0, 5, 10, 6, 1],
        "current_house_yrs": [11, 11, 13, 12, 12, 12],
        "job_stability": ["variable", "moderate", "variable", "variable", "moderate", "moderate"],
        "city_tier": ["unknown", "unknown", "unknown", "unknown", "unknown", "tier_3"],
    })

# Fixture to create y input Series for use in tests
@pytest.fixture
def y_input():
    return pd.Series([0, 1, 0, 0, 1, 0])


# --- TestStateDefaultRateTargetEncoder class ---
# Inherits from BaseSupervisedTransformerTests which adds the following tests:
# .test_fit_returns_self()
# .test_fit_learns_attributes()
# .test_instance_can_be_cloned()
# .test_fit_transform_equivalence()
# .test_transform_does_not_modify_input_df()
# .test_transform_handles_empty_df()
# .test_instance_can_be_pickled()
# .test_fit_raises_type_error_for_invalid_input()
# .test_transform_raises_type_error_for_invalid_input()
# .test_transform_raises_value_error_for_extra_column()
# .test_transform_raises_value_error_for_wrong_column_order()
# .test_transform_preserves_df_index()
# BaseSupervisedTransformerTests further inherits the following tests from BaseTransformerTests:
# .test_instantiation()
# .test_transform_raises_not_fitted_error_if_unfitted()
class TestStateDefaultRateTargetEncoder(BaseSupervisedTransformerTests):
    # Class instantiation 
    def test_instantiation(self, transformer):
        # First, run the .test_instantiation() method from the grandparent class BaseTransformerTests
        super().test_instantiation(transformer)
        # Then, add assertion specific to the StateDefaultRateTargetEncoder class
        assert isinstance(transformer, StateDefaultRateTargetEncoder)
    
    # Ensure .fit() raises ValueError if input DataFrame is missing the "state" column 
    @pytest.mark.unit
    def test_fit_raises_value_error_for_missing_state_column(self, transformer, X_input, y_input):
        X = X_input.copy()
        y = y_input.copy()
        X_without_state = X.drop(columns="state")
        expected_error_message = "Input X is missing the following columns: state."
        with pytest.raises(ValueError, match=expected_error_message):
            transformer.fit(X_without_state, y)

    # Ensure .fit() raises MissingValueError for missing values in the "state" column
    @pytest.mark.unit
    @pytest.mark.parametrize("missing_value", [None, np.nan])
    def test_fit_raises_missing_value_error_for_missing_states(self, transformer, missing_value):
        X_with_missing_state = pd.DataFrame({
            "state": ["state_1", missing_value, "state_2"]
        })
        y = pd.Series([0, 0, 1])  
        expected_error_message = "'state' column cannot contain missing values."
        with pytest.raises(MissingValueError, match=expected_error_message):
            transformer.fit(X_with_missing_state, y)

    # Ensure .fit() raises TypeError for non-string values in the "state" column
    @pytest.mark.unit
    @pytest.mark.parametrize("non_string_value", [
        ["a", "list"],
        ("a", "tuple"),
        {"a", "set"},
        {"a": "dictionary"}, 
        1,
        1.23,
        False
    ])
    def test_fit_raises_type_error_for_non_string_states(self, transformer, non_string_value):
        X_with_non_string_state = pd.DataFrame({
            "state": ["state_1", non_string_value, "state_2"]
        })  
        y = pd.Series([0, 0, 1])  
        expected_error_message = "All values in 'state' column must be strings."
        with pytest.raises(TypeError, match=expected_error_message):
            transformer.fit(X_with_non_string_state, y)    

    # Ensure .fit() raises TypeError if y input is not a pandas Series 
    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_y_input", [
        np.array([1, 2, 3]), 
        pd.DataFrame({"column_1": [1, 2, 3]}),
        "a string",
        ["a", "list"],
        ("a", "tuple"),
        {"a", "set"}, 
        {"a": "dictionary"},
        1,
        1.23,
        False,
        None
    ])
    def test_fit_raises_type_error_if_y_not_pandas_series(self, transformer, X_input, invalid_y_input):
        X = X_input.copy()
        expected_error_message = "Input y must be a pandas Series."
        with pytest.raises(TypeError, match=expected_error_message):
            transformer.fit(X, invalid_y_input)

    # Ensure .fit() raises MissingValueError for missing values on y input 
    @pytest.mark.unit
    @pytest.mark.parametrize("missing_value", [None, np.nan])
    def test_fit_raises_missing_value_error_for_missing_y_values(self, transformer, missing_value):
        X = pd.DataFrame({
            "state": ["state_1", "state_2", "state_3"]
        })
        y_with_missing_value = pd.Series([0, missing_value, 1])  
        expected_error_message = "Input y cannot contain missing values."
        with pytest.raises(MissingValueError, match=expected_error_message):
            transformer.fit(X, y_with_missing_value)
    
    # Ensure .fit() raises TypeError if y is not a pd.Series of integer type
    @pytest.mark.unit
    @pytest.mark.parametrize("y_with_wrong_type", [
        pd.Series([0.0, 1.0]), 
        pd.Series([True, False]), 
        pd.Series(["no default", "default"]), 
        pd.Series(["0", "1"]), 
    ])
    def test_fit_raises_type_error_if_y_not_integer(self, transformer, y_with_wrong_type):
        X = pd.DataFrame({
            "state": ["state_1", "state_2"]
        })
        expected_error_message = "Input y must be integer type."
        with pytest.raises(TypeError, match=expected_error_message):
            transformer.fit(X, y_with_wrong_type)

    # Ensure .fit() raises ValueError for y values that are not 0 or 1 
    @pytest.mark.unit
    @pytest.mark.parametrize("out_of_range_value", [-1, 2])
    def test_fit_raises_value_error_if_y_not_0_or_1(self, transformer, out_of_range_value):
        X = pd.DataFrame({
            "state": ["state_1", "state_2", "state_3"]
        })
        y_out_of_range = pd.Series([0, out_of_range_value, 1])  
        expected_error_message = re.escape("All y values must be 0 (no default) or 1 (default).")  # escape the string for the regex match
        with pytest.raises(ValueError, match=expected_error_message):
            transformer.fit(X, y_out_of_range)

    # Ensure .fit() raises ValueError for unequal index of X and y
    @pytest.mark.unit
    @pytest.mark.parametrize("X, y", [
        (
            pd.DataFrame({"state": ["state_1", "state_2", "state_3"]}, index=[1, 2, 3]), 
            pd.Series([0, 0, 1], index=[4, 5, 6])
        ),
        (
            pd.DataFrame({"state": ["state_1", "state_2", "state_3"]}, index=[1, 2, 3]), 
            pd.Series([0, 1], index=[1, 2])
        )
    ])
    def test_fit_raises_value_error_for_unequal_index_of_X_and_y(self, transformer, X, y):
        expected_error_message = "Input X and y must have the same index."
        with pytest.raises(ValueError, match=expected_error_message):
            transformer.fit(X, y)

    # Ensure .fit() correctly learns state default rates
    @pytest.mark.unit
    def test_fit_learns_state_default_rates(self, transformer):
        # X and y input
        X = pd.DataFrame({
            "state": ["state_1", "state_1", "state_2", "state_2"]
        })
        y = pd.Series([0, 1, 1, 1], name="default")
        # Fit
        transformer.fit(X, y)
        # Expected "default_rate_by_state_" learned attribute (pd.Series)
        expected_default_rate_by_state_index = pd.Index(["state_1", "state_2"], name="state")
        expected_default_rate_by_state_ = pd.Series([0.5, 1.0], index=expected_default_rate_by_state_index, name="default")
        # Ensure actual and expected "default_rate_by_state_" are identical
        assert_series_equal(transformer.default_rate_by_state_, expected_default_rate_by_state_)
    
    # Ensure .transform() raises ValueError if input DataFrame is missing the "state" column 
    @pytest.mark.unit
    def test_transform_raises_value_error_for_missing_state_column(self, transformer, X_input, y_input):
        X = X_input.copy()
        y = y_input.copy()
        X_without_state = X.drop(columns="state")
        # Fit on original DataFrame, but transform on DataFrame without "state" column 
        transformer.fit(X, y)
        expected_error_message = "Input X is missing the following columns: state."
        with pytest.raises(ValueError, match=expected_error_message):
            transformer.transform(X_without_state)  

    # Ensure .transform() raises MissingValueError for missing values in the "state" column
    @pytest.mark.unit
    @pytest.mark.parametrize("missing_value", [None, np.nan])
    def test_transform_raises_missing_value_error_for_missing_states(self, transformer, missing_value):
        X = pd.DataFrame({
            "state": ["state_1", "state_2", "state_3"]
        }) 
        y = pd.Series([0, 0, 1])
        X_with_missing_state = pd.DataFrame({
           "state": ["state_1", missing_value, "state_3"]
        })  
        # Fit on original DataFrame, but transform on DataFrame with missing state value 
        transformer.fit(X, y) 
        expected_error_message = "'state' column cannot contain missing values."
        with pytest.raises(MissingValueError, match=expected_error_message):
            transformer.transform(X_with_missing_state)

    # Ensure .transform() raises TypeError for non-string values in the "state" column
    @pytest.mark.unit
    @pytest.mark.parametrize("non_string_value", [
        ["a", "list"],
        ("a", "tuple"),
        {"a", "set"},
        {"a": "dictionary"}, 
        1,
        1.23,
        False
    ])
    def test_transform_raises_type_error_for_non_string_states(self, transformer, non_string_value):
        X = pd.DataFrame({
            "state": ["state_1", "state_2", "state_3"]
        }) 
        X_with_non_string_state = pd.DataFrame({
            "state": ["state_1", non_string_value, "state_3"]
        })  
        y = pd.Series([0, 0, 1])  
        # Fit on original DataFrame, but transform on DataFrame with non-string state 
        transformer.fit(X, y)
        expected_error_message = "All values in 'state' column must be strings."
        with pytest.raises(TypeError, match=expected_error_message):
            transformer.transform(X_with_non_string_state)

    # Ensure .transform() raises CategoricalLabelError for unknown states not seen during .fit()
    def test_transform_raises_categorical_label_error_for_unknown_states(self, transformer):
        X = pd.DataFrame({
            "state": ["state_1", "state_2", "state_3"]
        })
        y = pd.Series([0, 0, 1]) 
        X_with_unknown_state = pd.DataFrame({
            "state": ["state_1", "unknown_state", "state_3"]
        })  
        # Fit on original DataFrame, but transform on DataFrame with unknown state 
        transformer.fit(X, y)
        expected_error_message = "'state' column contains unknown states: unknown_state."
        with pytest.raises(CategoricalLabelError, match=expected_error_message):
            transformer.transform(X_with_unknown_state)

    # Ensure .transform() successfully adds the "state_default_rate" column 
    @pytest.mark.unit
    def test_transform_adds_state_default_rate_column(self, transformer):
        # X and y input
        X = pd.DataFrame({
            "state": ["state_1", "state_1", "state_2", "state_2"]
        })
        y = pd.Series([0, 1, 1, 1], name="default")
        # Fit and transform
        transformer.fit(X, y)
        X_transformed = transformer.transform(X)
        # Expected output
        expected_X_transformed = pd.DataFrame({
            "state": ["state_1", "state_1", "state_2", "state_2"],
            "state_default_rate": [0.5, 0.5, 1.0, 1.0]
        })
        # Ensure actual and expected output DataFrames are identical
        assert_frame_equal(X_transformed, expected_X_transformed)

    # Ensure .transform() ignores other columns 
    @pytest.mark.unit
    def test_transform_ignores_other_columns(self, transformer, X_input, y_input):
        X = X_input.copy()
        y = y_input.copy()
        # Create DataFrame of other columns
        other_columns = [column for column in X.columns if column != "state"]
        X_without_state = X[other_columns].copy()
        # Fit and transform on entire DataFrame
        transformer.fit(X, y)
        X_transformed = transformer.transform(X)
        # Create transformed DataFrame of other columns
        X_transformed_without_city = X_transformed[other_columns]
        # Ensure untransformed and transformed DataFrames of other columns are identical
        assert_frame_equal(X_without_state, X_transformed_without_city)
