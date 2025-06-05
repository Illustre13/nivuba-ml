# Standard library imports
import os
import sys
import re

# Third-party library imports
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
import numpy as np

# Add the root directory to the path for local imports (by going up two levels from current directory in which this file lives)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Local imports
from src.custom_transformers import BooleanColumnTransformer, ColumnMismatchError, MissingValueError, CategoricalLabelError
from src.global_constants import BOOLEAN_COLUMN_MAPPINGS
from tests.unit.base_transformer_tests import BaseTransformerTests


# --- Fixtures ---
# Fixture to instantiate BooleanColumnTransformer class for use in tests
@pytest.fixture
def transformer():
    return BooleanColumnTransformer(boolean_column_mappings=BOOLEAN_COLUMN_MAPPINGS)

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
        "profession": ["artist", "computer_hardware_engineer", "web_designer", "comedian", 
                       "financial_analyst", "statistician"],
        "city": ["sikar", "vellore", "bidar", "bongaigaon", "eluru[25]", "danapur"],
        "state": ["rajasthan", "tamil_nadu", "karnataka", "assam", "andhra_pradesh", "bihar"],
        "current_job_yrs": [3, 0, 5, 10, 6, 1],
        "current_house_yrs": [11, 11, 13, 12, 12, 12],
    })


# --- TestBooleanColumnTransformer class ---
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
class TestBooleanColumnTransformer(BaseTransformerTests):
    # Class instantiation 
    def test_instantiation(self, transformer):
        # First, run the .test_instantiation() method from the parent class BaseTransformerTests
        super().test_instantiation(transformer)
        # Then, add assertions specific to the BooleanColumnTransformer class
        assert isinstance(transformer, BooleanColumnTransformer)
        assert transformer.boolean_column_mappings == BOOLEAN_COLUMN_MAPPINGS

    # Ensure __init__() raises TypeError for invalid data types of "boolean_column_mappings" (must be a dictionary)
    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_boolean_column_mappings", [
        "a string",
        ["a", "list"],
        ("a", "tuple"),
        {"a", "set"},
        1,
        1.23,
        False,
        None
    ])
    def test_init_raises_type_error_for_invalid_boolean_column_mappings(self, invalid_boolean_column_mappings):
        expected_error_message = "'boolean_column_mappings' must be a dictionary specifying the mappings."
        with pytest.raises(TypeError, match=expected_error_message):
            BooleanColumnTransformer(boolean_column_mappings=invalid_boolean_column_mappings)
 
    # Ensure __init__() raises ValueError for empty "boolean_column_mappings" dictionary
    @pytest.mark.unit
    def test_init_raises_value_error_for_empty_boolean_column_mappings(self):
        expected_error_message = "'boolean_column_mappings' cannot be an empty dictionary. It must specify the the mappings."
        with pytest.raises(ValueError, match=expected_error_message):
            BooleanColumnTransformer(boolean_column_mappings={})

    # Ensure __init__() raises TypeError if any individual mapping within the "boolean_column_mappings" is not a dictionary
    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_mapping", [
        "a string",
        ["a", "list"],
        ("a", "tuple"),
        {"a", "set"},
        1,
        1.23,
        False,
        None
    ])
    def test_init_raises_type_error_for_invalid_mapping(self, invalid_mapping):
        expected_error_message = "The mapping for 'car_ownership' must be a dictionary."
        with pytest.raises(TypeError, match=expected_error_message):
            BooleanColumnTransformer(
                boolean_column_mappings={
                    "married": {"married": True, "single": False},
                    "car_ownership": invalid_mapping
                }
            )

    # Ensure __init__() raises ValueError if any value in the mappings is not boolean
    @pytest.mark.parametrize("non_boolean_value", [
        "a string",
        ["a", "list"],
        ("a", "tuple"),
        {"a": "dictionary"},
        {"a", "set"},
        1,
        1.23,
        None
    ])
    def test_init_raises_value_error_for_non_boolean_mapping_values(self, non_boolean_value):
        expected_error_message = re.escape("All values in the mapping for 'car_ownership' must be boolean (True or False).")
        with pytest.raises(ValueError, match=expected_error_message):
            BooleanColumnTransformer(
                boolean_column_mappings={
                    "married": {"married": True, "single": False},
                    "car_ownership": {"yes": True, "no": non_boolean_value}
                }
            )

    # Ensure .fit() raises ColumnMismatchError if any required column (from "boolean_column_mappings") is missing in X input DataFrame
    @pytest.mark.unit
    @pytest.mark.parametrize("missing_columns", [
        "married", 
        "car_ownership", 
        ["married", "car_ownership"],
    ])
    def test_fit_raises_column_mismatch_error_for_missing_columns(self, transformer, X_input, missing_columns):
        X = X_input.copy()
        X_with_missing_columns = X.drop(columns=missing_columns)
        with pytest.raises(ColumnMismatchError):
            transformer.fit(X_with_missing_columns)
    
    # Ensure .fit() raises MissingValueError for missing values in boolean columns
    @pytest.mark.unit
    @pytest.mark.parametrize("boolean_column", ["married", "car_ownership"])
    @pytest.mark.parametrize("missing_value", [None, np.nan])
    def test_fit_raises_missing_value_error_for_nan_in_boolean_columns(self, transformer, boolean_column, missing_value):
        X_with_missing = pd.DataFrame({
            "married": ["single", "married", "single"],
            "car_ownership": ["no", "yes", "no"],
        })
        X_with_missing.loc[0, boolean_column] = missing_value  # modify first row as a representative example
        expected_error_message = f"'{boolean_column}' column cannot contain missing values."
        with pytest.raises(MissingValueError, match=expected_error_message):
            transformer.fit(X_with_missing)
    
    # Ensure .fit() raises TypeError for invalid data types of boolean columns (must be str, int, float, bool)
    @pytest.mark.unit
    @pytest.mark.parametrize("boolean_column", ["married", "car_ownership"])
    @pytest.mark.parametrize("invalid_data_type", [
        ["a", "list"],
        ("a", "tuple"),
        {"a": "dictionary"},
        {"a", "set"}
    ])
    def test_fit_raises_type_error_for_invalid_boolean_column_type(self, transformer, boolean_column, invalid_data_type):
        X_with_invalid_type = pd.DataFrame({
            "married": ["single", "married", "single"],
            "car_ownership": ["no", "yes", "no"],
        })
        X_with_invalid_type.at[0, boolean_column] = invalid_data_type  # modify first row as a representative example
        expected_error_message = f"All values in '{boolean_column}' column must be str, int, float or bool."
        with pytest.raises(TypeError, match=expected_error_message):
            transformer.fit(X_with_invalid_type)

    # Ensure .fit() raises CategoricalLabelError for unknown labels (not in "boolean_column_mappings")
    @pytest.mark.unit
    @pytest.mark.parametrize("boolean_column, unknown_label", [
        ("married", "divorced"), 
        ("married", "yes"), 
        ("married", "no"),
        ("car_ownership", "maybe"),
        ("car_ownership", "lamborghini"),
        ("car_ownership", "soon"),
    ])
    def test_fit_raises_categorical_label_error_for_unknown_labels(self, transformer, boolean_column, unknown_label):
        X = pd.DataFrame({
            "married": ["single", "married", "single"],
            "car_ownership": ["no", "yes", "no"],
        })
        X.loc[0, boolean_column] = unknown_label  # modify first row as a representative example
        expected_error_message = f"'{boolean_column}' column contains unknown labels that are not in 'boolean_column_mappings': {unknown_label}."
        with pytest.raises(CategoricalLabelError, match=expected_error_message):
            transformer.fit(X)

    # Ensure .transform() successfully converts categorical string labels to boolean
    @pytest.mark.unit
    def test_transform_converts_string_categories_to_boolean(self, transformer, X_input):
        X = X_input.copy()
        # Fit and transform
        transformer.fit(X)
        X_transformed = transformer.transform(X)
        # Create expected output
        expected_X_transformed = X_input.copy()
        expected_X_transformed["married"] = [False, False, False, True, False, False]
        expected_X_transformed["car_ownership"] = [False, False, True, False, False, False]
        # Ensure transformed columns are boolean data type
        assert X_transformed["married"].dtype == "bool"
        assert X_transformed["car_ownership"].dtype == "bool"
        # Ensure actual and expected output DataFrames are identical
        assert_frame_equal(X_transformed, expected_X_transformed)

    # Ensure .transform() raises ColumnMismatchError if any required column (from "boolean_column_mappings") is missing in X input DataFrame
    @pytest.mark.unit
    @pytest.mark.parametrize("missing_columns", [
        "married", 
        "car_ownership", 
        ["married", "car_ownership"],
    ])
    def test_transform_raises_column_mismatch_error_for_missing_columns(self, transformer, X_input, missing_columns):
        X = X_input.copy()
        X_with_missing_columns = X.drop(columns=missing_columns)
        transformer.fit(X)
        with pytest.raises(ColumnMismatchError):
            transformer.transform(X_with_missing_columns)

    # Ensure .transform() raises MissingValueError for missing values in boolean columns
    @pytest.mark.unit
    @pytest.mark.parametrize("boolean_column", ["married", "car_ownership"])
    @pytest.mark.parametrize("missing_value", [None, np.nan])
    def test_transform_raises_missing_value_error_for_nan_in_boolean_columns(self, transformer, boolean_column, missing_value):
        X = pd.DataFrame({
            "married": ["single", "married", "single"],
            "car_ownership": ["no", "yes", "no"],
        })
        X_with_missing = X.copy()
        X_with_missing.loc[0, boolean_column] = missing_value  # modify first row as a representative example
        expected_error_message = f"'{boolean_column}' column cannot contain missing values."
        # Fit on original DataFrame, but transform on DataFrame with missing value
        transformer.fit(X)
        with pytest.raises(MissingValueError, match=expected_error_message):
            transformer.transform(X_with_missing)

    # Ensure .transform() raises TypeError for invalid data types of boolean columns (must be str, int, float, bool)
    @pytest.mark.unit
    @pytest.mark.parametrize("boolean_column", ["married", "car_ownership"])
    @pytest.mark.parametrize("invalid_data_type", [
        ["a", "list"],
        ("a", "tuple"),
        {"a": "dictionary"},
        {"a", "set"}
    ])
    def test_transform_raises_type_error_for_invalid_boolean_column_type(self, transformer, boolean_column, invalid_data_type):
        X = pd.DataFrame({
            "married": ["single", "married", "single"],
            "car_ownership": ["no", "yes", "no"],
        })
        X_with_invalid_type = X.copy()
        X_with_invalid_type.at[0, boolean_column] = invalid_data_type  # modify first row as a representative example
        expected_error_message = f"All values in '{boolean_column}' column must be str, int, float or bool."
        # Fit on original DataFrame, but transform on DataFrame with invalid data type
        transformer.fit(X)
        with pytest.raises(TypeError, match=expected_error_message):
            transformer.transform(X_with_invalid_type)

    # Ensure .transform() raises CategoricalLabelError for unknown labels (not in "boolean_column_mappings")
    @pytest.mark.unit
    @pytest.mark.parametrize("boolean_column, unknown_label", [
        ("married", "divorced"), 
        ("married", "yes"), 
        ("married", "no"),
        ("car_ownership", "maybe"),
        ("car_ownership", "lamborghini"),
        ("car_ownership", "soon"),
    ])
    def test_transform_raises_categorical_label_error_for_unknown_labels(self, transformer, boolean_column, unknown_label):
        X = pd.DataFrame({
            "married": ["single", "married", "single"],
            "car_ownership": ["no", "yes", "no"],
        })
        X_with_unknown_label = X.copy()
        X_with_unknown_label.loc[0, boolean_column] = unknown_label  # modify first row as a representative example
        expected_error_message = f"'{boolean_column}' column contains unknown labels that are not in 'boolean_column_mappings': {unknown_label}."
        # Fit on original DataFrame, but transform on DataFrame with unknown label
        transformer.fit(X)
        with pytest.raises(CategoricalLabelError, match=expected_error_message):
            transformer.transform(X_with_unknown_label)

    # Ensure .transform() ignores other columns not in "boolean_column_mappings"
    @pytest.mark.unit
    def test_transform_ignores_other_columns(self, transformer, X_input):
        X = X_input.copy()
        # Create DataFrame of other columns
        other_columns = [column for column in X.columns if column not in transformer.boolean_column_mappings]
        X_other_columns = X[other_columns].copy()
        # Fit and transform on entire DataFrame
        transformer.fit(X)
        X_transformed = transformer.transform(X)
        # Create transformed DataFrame of other columns
        X_transformed_non_boolean = X_transformed[other_columns]
        # Ensure untransformed and transformed DataFrames of other columns are identical
        assert_frame_equal(X_other_columns, X_transformed_non_boolean)

    # Ensure .transform() successfully maps integer categories to boolean
    @pytest.mark.unit
    def test_transform_converts_integer_categories_to_boolean(self):
        # Create DataFrame with integer categories
        X_integer_categories = pd.DataFrame({
            "married" : [0, 1, 0, 1, 0, 1],
            "car_ownership" : [5, 99, 5, 99, 5, 99],
            "current_job_yrs": [3, 0, 5, 10, 6, 1],
        })      
        # Instantiate and map integers to boolean
        transformer = BooleanColumnTransformer(boolean_column_mappings={
            "married": {1: True, 0: False},
            "car_ownership": {5: True, 99: False}              
        })
        # Fit and transform
        transformer.fit(X_integer_categories)
        X_transformed = transformer.transform(X_integer_categories)
        # Create expected output
        expected_X_transformed = pd.DataFrame({
            "married" : [False, True, False, True, False, True],
            "car_ownership" : [True, False, True, False, True, False],
            "current_job_yrs": [3, 0, 5, 10, 6, 1],
        }) 
        # Ensure transformed columns are boolean data type
        assert X_transformed["married"].dtype == "bool"
        assert X_transformed["car_ownership"].dtype == "bool"
        # Ensure actual and expected output DataFrames are identical
        assert_frame_equal(X_transformed, expected_X_transformed)
