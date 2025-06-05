# Standard library imports
import os
import sys

# Third-party library imports
import pytest
import pandas as pd
import numpy as np

# Add the root directory to the path for local imports (by going up two levels from current directory in which this file lives)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Local imports
from src.custom_transformers import MissingValueChecker, MissingValueError, ColumnMismatchError
from src.global_constants import CRITICAL_FEATURES, NON_CRITICAL_FEATURES
from tests.unit.base_transformer_tests import BaseTransformerTests


# --- Fixtures ---
# Fixture to instantiate MissingValueChecker class for use in tests
@pytest.fixture
def transformer():
    return MissingValueChecker(
        critical_features=CRITICAL_FEATURES, 
        non_critical_features=NON_CRITICAL_FEATURES
    )

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


# --- TestMissingValueChecker class ---
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
class TestMissingValueChecker(BaseTransformerTests):
    # Class instantiation 
    @pytest.mark.unit
    def test_instantiation(self, transformer):
        # First, run the .test_instantiation() method from the parent class BaseTransformerTests
        super().test_instantiation(transformer)
        # Then, add assertions specific to the MissingValueChecker class
        assert isinstance(transformer, MissingValueChecker)
        assert transformer.critical_features == CRITICAL_FEATURES
        assert transformer.non_critical_features == NON_CRITICAL_FEATURES

    # Ensure __init__() raises TypeError for invalid data types of "critical_features" (must be a list)
    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_critical_features", [
        "a string",
        {"a": "dictionary"},
        ("a", "tuple"),
        1,
        1.23,
        False,
        None        
    ])
    def test_init_raises_type_error_for_invalid_critical_features(self, invalid_critical_features):
        expected_error_message = "'critical_features' must be a list of column names."
        with pytest.raises(TypeError, match=expected_error_message):
            MissingValueChecker(invalid_critical_features, NON_CRITICAL_FEATURES)

    # Ensure __init__() raises TypeError for invalid data types of "non_critical_features" (must be a list)
    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_non_critical_features", [
        "a string",
        {"a": "dictionary"},
        ("a", "tuple"),
        1,
        1.23,
        False,
        None        
    ])
    def test_init_raises_type_error_for_invalid_non_critical_features(self, invalid_non_critical_features):
        expected_error_message = "'non_critical_features' must be a list of column names."
        with pytest.raises(TypeError, match=expected_error_message):
            MissingValueChecker(CRITICAL_FEATURES, invalid_non_critical_features)

    # Ensure __init__() raises ValueError for empty "critical_features" list
    @pytest.mark.unit
    def test_init_raises_value_error_for_empty_critical_features(self):
        expected_error_message = "'critical_features' cannot be an empty list. It must specify the names of the critical features."
        with pytest.raises(ValueError, match=expected_error_message):
            MissingValueChecker(critical_features=[], non_critical_features=NON_CRITICAL_FEATURES)

    # Ensure __init__() raises ValueError for empty "non_critical_features" list
    @pytest.mark.unit
    def test_init_raises_value_error_for_empty_non_critical_features(self):
        expected_error_message = "'non_critical_features' cannot be an empty list. It must specify the names of the non-critical features."
        with pytest.raises(ValueError, match=expected_error_message):
            MissingValueChecker(critical_features=CRITICAL_FEATURES, non_critical_features=[])

    # Ensure .fit() raises ColumnMismatchError for missing columns in the input DataFrame
    @pytest.mark.unit
    @pytest.mark.parametrize("missing_columns", [
        "income", 
        "age", 
        ["experience", "married"],
        ["house_ownership", "car_ownership"],
        ["profession", "city", "state"],
        ["current_job_yrs", "current_house_yrs"],
    ])
    def test_fit_raises_column_mismatch_error_for_missing_columns(self, transformer, X_input, missing_columns):
        X = X_input.copy()
        X_with_missing_columns = X.drop(columns=missing_columns)
        with pytest.raises(ColumnMismatchError):
            transformer.fit(X_with_missing_columns)

    # Ensure .fit() raises ColumnMismatchError for unexpected columns in the input DataFrame
    @pytest.mark.unit
    @pytest.mark.parametrize("unexpected_columns", [
        ["unexpected_column_1"],
        ["unexpected_column_1", "unexpected_column_2"],
        ["unexpected_column_1", "unexpected_column_2", "unexpected_column_3"]
    ])
    def test_fit_raises_column_mismatch_error_for_unexpected_columns(self, transformer, X_input, unexpected_columns):
        X_with_unexpected_columns = X_input.copy()
        for unexpected_column in unexpected_columns:
            X_with_unexpected_columns[unexpected_column] = 5 
        with pytest.raises(ColumnMismatchError):
            transformer.fit(X_with_unexpected_columns)

    # Ensure .fit() raises MissingValueError for missing values in critical features
    @pytest.mark.unit
    @pytest.mark.parametrize("missing_value", [None, np.nan])
    @pytest.mark.parametrize("critical_feature", CRITICAL_FEATURES)
    def test_fit_raises_missing_value_error_for_critical_features(self, transformer, X_input, missing_value, critical_feature):
        X_with_missing_value = X_input.copy()
        X_with_missing_value.loc[0:2, critical_feature] = missing_value  # missings on rows 0-2 as an example
        # Create expected dictionary of number of missing values by column 
        expected_missing_by_column_dict = {"income": 0, "age": 0, "experience": 0, "profession": 0, "city": 0, "state": 0, "current_job_yrs": 0, "current_house_yrs": 0}
        expected_missing_by_column_dict[critical_feature] = 3  # 3 missing values
        # Create expected error message text
        expected_error_message = (
            f"3 missing values found in critical features "
            f"across 3 rows. Please provide missing values.\n"
            f"Missing values by column: {expected_missing_by_column_dict}" 
        )
        with pytest.raises(MissingValueError, match=expected_error_message):
            transformer.fit(X_with_missing_value)

    # Ensure .fit() prints warning message for missing values in non-critical features
    @pytest.mark.unit
    @pytest.mark.parametrize("missing_value", [None, np.nan])
    @pytest.mark.parametrize("non_critical_feature", NON_CRITICAL_FEATURES)
    def test_fit_prints_missing_value_warning_message_for_non_critical_features(self, transformer, X_input, missing_value, non_critical_feature, capsys):
        X_with_missing_value = X_input.copy()
        X_with_missing_value.loc[0:2, non_critical_feature] = missing_value  
        transformer.fit(X_with_missing_value)
        # Create expected dictionary of number of missing values by column 
        expected_missing_by_column_dict = {"married": 0, "car_ownership": 0, "house_ownership": 0}
        expected_missing_by_column_dict[non_critical_feature] = 3  
        # Create expected warning message text
        expected_warning_message = (
            f"Warning: 3 missing values found in non-critical features "
            f"across 3 rows. Missing values will be imputed.\n"
            f"Missing values by column: {expected_missing_by_column_dict}\n" 
        )
        # Capture standard output and standard error
        captured_output_and_error = capsys.readouterr()
        # Ensure standard output is the expected warning message
        assert captured_output_and_error.out == expected_warning_message
        # Ensure nothing was written to standard error
        assert captured_output_and_error.err == ""

    # Ensure .fit() raises MissingValueError for missing values in both critical and non-critical features
    @pytest.mark.unit
    def test_fit_raises_missing_value_error_for_critical_and_non_critical_missing(self, transformer, X_input):
        # Use representative examples
        example_critical_feature = CRITICAL_FEATURES[0]
        example_non_critical_feature = NON_CRITICAL_FEATURES[0]
        # Create DataFrame with missing value on both a critical and non-critical feature
        X_with_both_missing = X_input.copy()
        X_with_both_missing[example_critical_feature] = np.nan  
        X_with_both_missing[example_non_critical_feature] = np.nan
        # Ensure .fit() raises MissingValueError
        with pytest.raises(MissingValueError):
            transformer.fit(X_with_both_missing)

    # Ensure .fit() raises MissingValueError for non-critical feature with only missing values
    @pytest.mark.unit
    @pytest.mark.parametrize("non_critical_feature", NON_CRITICAL_FEATURES)
    def test_fit_raises_missing_value_error_for_non_critical_feature_with_only_missings(self, transformer, X_input, non_critical_feature):
        # Create DataFrame with a non-critical feature with only missing values
        X_with_only_missings = X_input.copy()
        X_with_only_missings[non_critical_feature] = np.nan
        # Ensure .fit() raises MissingValueError
        expected_error_message = f"'{non_critical_feature}' cannot be only missing values. Please ensure at least one non-missing value."
        with pytest.raises(MissingValueError, match=expected_error_message):
            transformer.fit(X_with_only_missings)

    # Ensure .transform() raises ColumnMismatchError for missing columns in the input DataFrame
    @pytest.mark.unit
    @pytest.mark.parametrize("missing_columns", [
        "income", 
        "age", 
        ["experience", "married"],
        ["house_ownership", "car_ownership"],
        ["profession", "city", "state"],
        ["current_job_yrs", "current_house_yrs"],
    ])
    def test_transform_raises_column_mismatch_error_for_missing_columns(self, transformer, X_input, missing_columns):
        X = X_input.copy()
        X_with_missing_columns = X.drop(columns=missing_columns)
        # Fit on original DataFrame, but transform on DataFrame with missing columns
        transformer.fit(X)
        with pytest.raises(ColumnMismatchError):
            transformer.transform(X_with_missing_columns)

    # Override parent class method to ensure unexpected columns raise ColumnMismatchError instead of ValueError (see next method)
    def test_transform_raises_value_error_for_extra_column(self):
        pass
    
    # Ensure .transform() raises ColumnMismatchError for unexpected columns in the input DataFrame
    @pytest.mark.unit
    @pytest.mark.parametrize("unexpected_columns", [
        ["unexpected_column_1"],
        ["unexpected_column_1", "unexpected_column_2"],
        ["unexpected_column_1", "unexpected_column_2", "unexpected_column_3"]
    ])
    def test_transform_raises_column_mismatch_error_for_unexpected_columns(self, transformer, X_input, unexpected_columns):
        X = X_input.copy()
        X_with_unexpected_columns = X_input.copy()
        for unexpected_column in unexpected_columns:
            X_with_unexpected_columns[unexpected_column] = 5 
        # Fit on original DataFrame, but transform on DataFrame with unexpected columns
        transformer.fit(X)
        with pytest.raises(ColumnMismatchError):
            transformer.transform(X_with_unexpected_columns)

    # Ensure .transform() raises MissingValueError for missing values in critical features
    @pytest.mark.unit
    @pytest.mark.parametrize("missing_value", [None, np.nan])
    @pytest.mark.parametrize("critical_feature", CRITICAL_FEATURES)
    def test_transform_raises_missing_value_error_for_critical_features(self, transformer, X_input, missing_value, critical_feature):
        X = X_input.copy()
        transformer.fit(X)
        X_with_missing_value = X_input.copy()
        X_with_missing_value.loc[0:2, critical_feature] = missing_value  # missings on rows 0-2 as an example
        # Create expected dictionary of number of missing values by column 
        expected_missing_by_column_dict = {"income": 0, "age": 0, "experience": 0, "profession": 0, "city": 0, "state": 0, "current_job_yrs": 0, "current_house_yrs": 0}
        expected_missing_by_column_dict[critical_feature] = 3  # 3 missing values
        # Create expected error message text
        expected_error_message = (
            f"3 missing values found in critical features "
            f"across 3 rows. Please provide missing values.\n"
            f"Missing values by column: {expected_missing_by_column_dict}" 
        )
        with pytest.raises(MissingValueError, match=expected_error_message):
            transformer.transform(X_with_missing_value)

    # Ensure .transform() prints warning message for missing values in non-critical features
    @pytest.mark.unit
    @pytest.mark.parametrize("missing_value", [None, np.nan])
    @pytest.mark.parametrize("non_critical_feature", NON_CRITICAL_FEATURES)
    def test_transform_prints_missing_value_warning_message_for_non_critical_features(self, transformer, X_input, missing_value, non_critical_feature, capsys):
        X = X_input.copy()
        transformer.fit(X)
        X_with_missing_value = X_input.copy()
        X_with_missing_value.loc[0:2, non_critical_feature] = missing_value  
        transformer.transform(X_with_missing_value)
        # Create expected dictionary of number of missing values by column 
        expected_missing_by_column_dict = {"married": 0, "car_ownership": 0, "house_ownership": 0}
        expected_missing_by_column_dict[non_critical_feature] = 3  
        # Create expected warning message text
        expected_warning_message = (
            f"Warning: 3 missing values found in non-critical features "
            f"across 3 rows. Missing values will be imputed.\n"
            f"Missing values by column: {expected_missing_by_column_dict}\n" 
        )
        # Capture standard output and standard error
        captured_output_and_error = capsys.readouterr()
        # Ensure standard output is the expected warning message
        assert captured_output_and_error.out == expected_warning_message
        # Ensure nothing was written to standard error
        assert captured_output_and_error.err == ""

    # Ensure .transform() raises MissingValueError for missing values in both critical and non-critical features
    @pytest.mark.unit
    def test_transform_raises_missing_value_error_for_critical_and_non_critical_missing(self, transformer, X_input):
        X = X_input.copy()
        transformer.fit(X)
        # Use representative examples
        example_critical_feature = CRITICAL_FEATURES[0]
        example_non_critical_feature = NON_CRITICAL_FEATURES[0]
        # Create DataFrame with missing value on both a critical and non-critical feature
        X_with_both_missing = X_input.copy()
        X_with_both_missing[example_critical_feature] = np.nan  
        X_with_both_missing[example_non_critical_feature] = np.nan
        # Ensure .transform() raises MissingValueError
        with pytest.raises(MissingValueError):
            transformer.transform(X_with_both_missing)
