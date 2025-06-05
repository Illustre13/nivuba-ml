# Standard library imports
import os
import sys

# Third-party library imports
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

# Add the root directory to the path for local imports (by going up two levels from current directory in which this file lives)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Local imports
from src.custom_transformers import FeatureSelector, ColumnMismatchError
from src.global_constants import COLUMNS_TO_KEEP
from tests.unit.base_transformer_tests import BaseTransformerTests


# --- Fixtures ---
# Fixture to instantiate FeatureSelector class for use in tests
@pytest.fixture
def transformer():
    return FeatureSelector(columns_to_keep=COLUMNS_TO_KEEP)

# Fixture to create X input DataFrame for use in tests
@pytest.fixture
def X_input():
    return pd.DataFrame({
      "income": [-0.82, 1.55, -1.47, 0.51, 1.39, -0.61],
      "age": [-0.64, -0.52, -0.29, -0.52, 0.23, 0.46],
      "experience": [-1.68, -0.84, -1.51, -1.68, 1.31, -0.18],
      "current_job_yrs": [-1.73, -0.36, -1.46, -1.73, -0.36, 0.73],
      "current_house_yrs": [-0.71, 0.71, -0.71, 1.43, 0.71, -0.71],
      "state_default_rate": [-1.08, -1.24, 0.32, -0.31, -0.14, -0.38],
      "house_ownership_owned": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
      "house_ownership_rented": [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
      "job_stability": [1.0, 0.0, 2.0, 3.0, 0.0, 3.0],
      "city_tier": [0.0, 2.0, 0.0, 1.0, 0.0, 3.0],
      "married": [False, False, True, False, True, False],
      "car_ownership": [False, True, False, False, True, True],
      "profession": ["computer_hardware_engineer", "web_designer", "lawyer", "firefighter", "artist", "librarian"],
      "city": ["vellore", "bidar", "nizamabad", "farrukhabad", "sikar", "hindupur"],
      "state": ["tamil_nadu", "karnataka", "telangana", "uttar_pradesh", "rajasthan", "andhra_pradesh"]
    })


# --- TestFeatureSelector class ---
# Inherits from BaseTransformerTests which adds the following tests:
# .test_instantiation()
# .test_fit_returns_self()
# .test_fit_learns_attributes()
# .test_instance_can_be_cloned()
# .test_fit_transform_equivalence()
# .test_transform_does_not_modify_input_df()
# .test_transform_handles_empty_df()
# .test_instance_can_be_pickled()
# .test_fit_raises_type_error_for_invalid_input()
# .test_transform_raises_not_fitted_error_if_unfitted()
# .test_transform_raises_type_error_for_invalid_input()
# .test_transform_raises_value_error_for_extra_column()
# .test_transform_raises_value_error_for_wrong_column_order()
# .test_transform_preserves_df_index()
class TestFeatureSelector(BaseTransformerTests):
    # Class instantiation 
    @pytest.mark.unit
    def test_instantiation(self, transformer):
        # First, run the .test_instantiation() method from the parent class BaseTransformerTests
        super().test_instantiation(transformer)  # asserts transformer is BaseEstimator and TransformerMixin
        # Then, add assertions specific to the FeatureSelector class
        assert isinstance(transformer, FeatureSelector)
        assert transformer.columns_to_keep == COLUMNS_TO_KEEP

    # Ensure __init__() raises TypeError for invalid data types of "columns_to_keep" (must be a list)
    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_columns_to_keep", [
        "a string",
        ("a", "tuple"),
        {"a": "dictionary"},
        {"a", "set"}, 
        1,
        1.23,
        False,
        None
    ])
    def test_init_raises_type_error_for_invalid_columns_to_keep(self, invalid_columns_to_keep):
        expected_error_message = "'columns_to_keep' must be a list of column names."
        with pytest.raises(TypeError, match=expected_error_message):
            FeatureSelector(invalid_columns_to_keep)

    # Ensure __init__() raises ValueError for empty "columns_to_keep" list
    @pytest.mark.unit
    def test_init_raises_value_error_for_empty_columns_to_keep(self):
        expected_error_message = "'columns_to_keep' cannot be an empty list. It must specify the column names."
        with pytest.raises(ValueError, match=expected_error_message):
            FeatureSelector(columns_to_keep=[])

    # Ensure .fit() raises ColumnMismatchError for missing columns in the input DataFrame
    @pytest.mark.unit
    @pytest.mark.parametrize("missing_columns", [
        "income", 
        "age", 
        "experience",
        ["current_job_yrs", "current_house_yrs"],
        ["state_default_rate", "house_ownership_owned", "house_ownership_rented"],
        ["job_stability", "city_tier", "married", "car_ownership"]
    ])
    def test_fit_raises_column_mismatch_error_for_missing_columns(self, transformer, X_input, missing_columns):
        X = X_input.copy()
        X_with_missing_columns = X.drop(columns=missing_columns)
        with pytest.raises(ColumnMismatchError):
            transformer.fit(X_with_missing_columns)

    # Ensure .fit() ignores extra columns not in columns_to_keep
    @pytest.mark.unit
    def test_fit_ignores_extra_columns(self, transformer, X_input):
        X = X_input.copy()
        X["extra_column"] = "extra_value"  # extra column that is not in COLUMNS_TO_KEEP
        transformer.fit(X)  # should fit without raising an error
        # Ensure the learned feature number and names are same as in input DataFrame
        assert transformer.n_features_in_ == X.shape[1]
        assert transformer.feature_names_in_ == X.columns.tolist()

    # Ensure .transform() removes other columns not specified in "columns_to_keep"
    @pytest.mark.unit
    def test_transform_removes_other_columns(self):
        X = pd.DataFrame({
            "column_to_keep_1": [1, 2, 3],
            "column_to_keep_2": [1, 2, 3],
            "column_to_keep_3": [1, 2, 3],
            "column_to_remove_1": [1, 2, 3],
            "column_to_remove_2": [1, 2, 3],
        })
        # Instantiate, fit and transform
        transformer = FeatureSelector(columns_to_keep=["column_to_keep_1", "column_to_keep_2", "column_to_keep_3"])
        transformer.fit(X)
        X_transformed = transformer.transform(X)
        # Expected output 
        expected_X_transformed = pd.DataFrame({
            "column_to_keep_1": [1, 2, 3],
            "column_to_keep_2": [1, 2, 3],
            "column_to_keep_3": [1, 2, 3],
        })
        # Ensure actual and expected output DataFrames are identical
        assert_frame_equal(X_transformed, expected_X_transformed)

    # Ensure .transform() raises ValueError for missing columns in the input DataFrame
    @pytest.mark.unit
    @pytest.mark.parametrize("missing_columns", [
        "income", 
        "age", 
        "experience",
        ["current_job_yrs", "current_house_yrs"],
        ["state_default_rate", "house_ownership_owned", "house_ownership_rented"],
        ["job_stability", "city_tier", "married", "car_ownership"]
    ])
    def test_transform_raises_value_error_for_missing_columns(self, transformer, X_input, missing_columns):
        X = X_input.copy()
        transformer.fit(X)
        X_with_missing_columns = X.drop(columns=missing_columns)
        with pytest.raises(ValueError):
            transformer.transform(X_with_missing_columns)

    # Ensure .transform() preserves data types of columns to keep
    @pytest.mark.unit
    @pytest.mark.parametrize("column, values", [
        ("integer_column", [1, 2]),
        ("float_column", [1.1, 2.2]),
        ("boolean_column", [True, False]),
        ("string_column", ["string", "values"]),
    ])
    def test_transform_preserves_data_types(self, column, values):
        X = pd.DataFrame({
            column: values,
            "column_to_remove": ["some", "values"],
        })
        # Instantiate, fit and transform
        transformer = FeatureSelector(columns_to_keep=[column])
        transformer.fit(X)
        X_transformed = transformer.transform(X)
        # Ensure column has same data type in transformed and original DataFrame
        assert X_transformed[column].dtype == X[column].dtype
