# Standard library imports
import os
import sys

# Third-party library imports
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np

# Add the root directory to the path for local imports (by going up two levels from current directory in which this file lives)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Local imports
from src.custom_transformers import SnakeCaseFormatter, ColumnMismatchError
from src.global_constants import COLUMNS_FOR_SNAKE_CASING
from tests.unit.base_transformer_tests import BaseTransformerTests


# --- Fixtures ---
# Fixture to instantiate CategoricalLabelStandardizer class for use in tests
@pytest.fixture
def transformer():
    return SnakeCaseFormatter(columns=COLUMNS_FOR_SNAKE_CASING)

# Fixture to create X input DataFrame for use in tests
@pytest.fixture
def X_input():
    return pd.DataFrame({
        "income": [9121364, 2636544, 9470213, 6558967, 6245331, 154867],
        "age": [70, 39, 41, 41, 65, 64],
        "experience": [18, 0, 5, 10, 6, 1],
        "married": ["single", "single", "single", "married", "single", "single"],
        "house_ownership": ["rented", "rented", "norent_noown", "rented", "rented", "owned"],
        "car_ownership": ["no", "no", "yes", "no", "no", "no"],
        "profession": ["Artist", "Computer_hardware_engineer", "Web_designer", "Comedian", 
                       "Financial_Analyst", "Statistician"],
        "city": ["Sikar", "Vellore", "Bidar", "Bongaigaon", "Eluru[25]", "Danapur"],
        "state": ["Rajasthan", "Tamil_Nadu", "Karnataka", "Assam", "Andhra_Pradesh", "Bihar"],
        "current_job_yrs": [3, 0, 5, 10, 6, 1],
        "current_house_yrs": [11, 11, 13, 12, 12, 12],
    })


# --- TestCategoricalLabelStandardizer class ---
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
class TestSnakeCaseFormatter(BaseTransformerTests):
    # Class instantiation 
    @pytest.mark.unit
    def test_instantiation(self, transformer):
        # First, run the .test_instantiation() method from the parent class BaseTransformerTests
        super().test_instantiation(transformer)
        # Then, add assertions specific to the SnakeCaseFormatter class
        assert isinstance(transformer, SnakeCaseFormatter)
        assert transformer.columns == COLUMNS_FOR_SNAKE_CASING

    # Ensure __init__() raises TypeError for invalid data types of "columns" (must be a list or None)
    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_columns", [
        "a string",
        {"a": "dictionary"},
        ("a", "tuple"),
        1,
        1.23,
        False
    ])
    def test_init_raises_type_error_for_invalid_columns(self, invalid_columns):
        expected_error_message = "'columns' must be a list of column names or None. If None, all columns will be used."
        with pytest.raises(TypeError, match=expected_error_message):
            SnakeCaseFormatter(columns=invalid_columns)

    # Ensure __init__() raises ValueError for empty "columns" list
    @pytest.mark.unit
    def test_init_raises_value_error_for_empty_columns(self):
        expected_error_message = "'columns' cannot be an empty list. It must specify the column names for snake case formatting."
        with pytest.raises(ValueError, match=expected_error_message):
            SnakeCaseFormatter(columns=[])

    # Ensure .fit() raises ColumnMismatchError for missing columns in the input DataFrame
    @pytest.mark.unit
    @pytest.mark.parametrize("missing_columns", [
        "profession", 
        "city", 
        "state", 
        ["profession", "city"],
        ["profession", "city", "state"],
    ])
    def test_fit_raises_column_mismatch_error_for_missing_columns(self, transformer, X_input, missing_columns):
        X = X_input.copy()
        X_with_missing_columns = X.drop(columns=missing_columns)
        with pytest.raises(ColumnMismatchError):
            transformer.fit(X_with_missing_columns)

    # Ensure .fit() ignores extra columns not provided in columns hyperparameter
    @pytest.mark.unit
    def test_fit_ignores_extra_column(self, transformer, X_input):
        X_with_extra_column = X_input.copy()
        # Add extra column that is not in COLUMNS_FOR_SNAKE_CASING
        X_with_extra_column["extra_column"] = "extra_value" 
        # .fit() should not raise an error 
        transformer.fit(X_with_extra_column)  
        # Ensure the learned feature number and names are same as in input DataFrame
        assert transformer.n_features_in_ == X_with_extra_column.shape[1]
        assert transformer.feature_names_in_ == X_with_extra_column.columns.tolist()
    
    # Ensure .transform() raises ValueError for extra column not seen during .fit()
    @pytest.mark.unit
    def test_transform_raises_value_error_for_unexpected_column(self, transformer, X_input):
        X = X_input.copy()
        X_with_extra_column = X_input.copy()
        X_with_extra_column["extra_column"] = "extra_value"   
        # Fit with extra column
        transformer.fit(X_with_extra_column)
        # Ensure .transform() without extra column raises ValueError
        with pytest.raises(ValueError, match="Feature names and feature order of input X must be the same as during .fit()."):
            transformer.transform(X)

    # Ensure .transform() formats single string column in snake case
    @pytest.mark.unit
    @pytest.mark.parametrize("input_value, expected_output_value", [
        (" leading space", "leading_space"),
        ("trailing space ", "trailing_space"),
        ("   three leading spaces and two trailing spaces  ", "three_leading_spaces_and_two_trailing_spaces"),
        ("Title Case With Inner Spaces", "title_case_with_inner_spaces"),
        ("lower-case-with-hyphens", "lower_case_with_hyphens"),
        ("Title/Case/With/Slashes", "title_case_with_slashes"),
        ("ALL-CAPS/AND SLASH", "all_caps_and_slash"),
        ("  Leading spaces and Mixed Case and-hypen and/slash", "leading_spaces_and_mixed_case_and_hypen_and_slash"),
        ("already_in_snake_case", "already_in_snake_case"),
        ("", ""),
        ("   ", ""),
        ("multiple--hypens and//slashes and inner   spaces", "multiple__hypens_and__slashes_and_inner___spaces"),
        ("version 2.0!", "version_2.0!"),
    ])
    def test_transform_formats_single_string_column_in_snake_case(self, transformer, X_input, input_value, expected_output_value):
        X = X_input.copy()
        # Modify first row of "city" column as a representative example
        X.loc[0, "city"] = input_value 
        # Fit and transform
        transformer.fit(X)
        X_transformed = transformer.transform(X)
        # Ensure value is formatted in snake case
        assert X_transformed.loc[0, "city"] == expected_output_value
  
    # Ensure .transform() formats multiple string columns in snake case
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "columns_for_snake_casing, expected_output",
        [
            # Format two columns in snake case
            (
                ["profession", "city"],
                {
                    "profession": ["some_profession"],
                    "city": ["some_city"],
                    "state": ["Some State"],
                    "income": [100000],
                },
            ),
            # Format all string columns in snake case by default (when columns=None)
            (
                None,
                {
                    "profession": ["some_profession"],
                    "city": ["some_city"],
                    "state": ["some_state"],
                    "income": [100000],
                },
            ),
        ],
    )
    def test_transform_formats_multiple_string_columns_in_snake_case(self, columns_for_snake_casing, expected_output):
        transformer = SnakeCaseFormatter(columns=columns_for_snake_casing)
        X = pd.DataFrame({
            "profession": ["Some Profession"],
            "city": ["Some City"],
            "state": ["Some State"],
            "income": [100000],
        })
        expected_X_transformed = pd.DataFrame(expected_output)
        # Fit and transform
        transformer.fit(X)
        X_transformed = transformer.transform(X)
        # Ensure actual and expected output DataFrame are identical
        assert_frame_equal(X_transformed, expected_X_transformed)
      
    # Ensure .transform() ignores non-string data types
    @pytest.mark.unit
    @pytest.mark.parametrize("non_string_value", [
        1,
        1.23,
        False,
        None,
        np.nan,
        ["a", "list"],
        ("a", "tuple"),
        {"a": "dict"}
    ])
    def test_transform_ignores_non_string_types(self, transformer, non_string_value):
        # Input DataFrame
        X = pd.DataFrame({
            "profession": ["First Profession", "Second Profession"],
            "city": ["First City", non_string_value],
            "state": ["First State", "Second State"]
        })
        # Expected output DataFrame
        expected_X_transformed = pd.DataFrame({
            "profession": ["first_profession", "second_profession"],
            "city": ["first_city", non_string_value],
            "state": ["first_state", "second_state"]
        })
        # Fit and transform
        transformer.fit(X) 
        X_transformed = transformer.transform(X)
        # Ensure actual and expected output DataFrame are identical
        assert_frame_equal(X_transformed, expected_X_transformed)
