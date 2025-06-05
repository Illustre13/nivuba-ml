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
from src.pipeline import create_missing_value_handling_pipeline
from src.custom_transformers import MissingValueError, ColumnMismatchError
from src.global_constants import CRITICAL_FEATURES, NON_CRITICAL_FEATURES
from tests.integration.base_pipeline_tests import BasePipelineTests


# --- Fixtures ---
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

# Fixture to create the missing value handling pipeline segment for use in tests
@pytest.fixture
def pipeline(): 
    return create_missing_value_handling_pipeline()


# --- TestMissingValueHandlingPipeline class ---
# Inherits from BasePipelineTests which adds the following integration tests:
# .test_pipeline_can_be_cloned()
# .test_pipeline_fit_transform_equivalence()
# .test_pipeline_fit_and_transform_raise_type_error_if_X_not_df()
# .test_pipeline_transform_raises_not_fitted_error_if_unfitted()
# .test_pipeline_transform_does_not_modify_input_df()
# .test_fitted_pipeline_can_be_pickled()
# .test_pipeline_transform_raises_value_error_for_wrong_column_order()
# .test_pipeline_transform_preserves_df_index()
# .test_pipeline_transform_passes_through_empty_df()
class TestMissingValueHandlingPipeline(BasePipelineTests):
    # Ensure pipeline .fit() and .transform() raise MissingValueError for missing values in critical features
    @pytest.mark.integration
    @pytest.mark.parametrize("method", ["fit", "transform"])
    @pytest.mark.parametrize("missing_value", [None, np.nan, pd.NA])
    @pytest.mark.parametrize("critical_feature", CRITICAL_FEATURES)
    def test_missing_value_handling_pipeline_raises_missing_value_error_for_critical_features(self, X_input, pipeline, method, missing_value, critical_feature):
        X = X_input.copy()
        X_with_missing_values = X_input.copy()
        X_with_missing_values[critical_feature] = missing_value
        # Create expected dictionary of number of missing values by column 
        expected_missing_by_column_dict = {"income": 0, "age": 0, "experience": 0, "profession": 0, "city": 0, "state": 0, "current_job_yrs": 0, "current_house_yrs": 0}
        expected_missing_by_column_dict[critical_feature] = 6  # X_input has 6 rows
        # Create expected error message text
        expected_error_message = (
            f"6 missing values found in critical features "
            f"across 6 rows. Please provide missing values.\n"
            f"Missing values by column: {expected_missing_by_column_dict}" 
        )
        # Ensure .fit() raises MissingValueError with expected error message text
        if method == "fit":
            with pytest.raises(MissingValueError, match=expected_error_message):
                pipeline.fit(X_with_missing_values)
        # Ensure .transform() raises MissingValueError with expected error message text
        else:
            # Fit on original DataFrame, but transform on DataFrame with missing values
            pipeline.fit(X)
            with pytest.raises(MissingValueError, match=expected_error_message):
                pipeline.transform(X_with_missing_values)

    # Ensure pipline .fit() prints warning message and learns mode for missing values in non-critical features
    @pytest.mark.integration
    @pytest.mark.parametrize("missing_value", [None, np.nan, pd.NA])
    @pytest.mark.parametrize("non_critical_feature", NON_CRITICAL_FEATURES)
    def test_missing_value_handling_pipeline_fit_warns_and_learns_mode_for_missing_values_in_non_critical_features(self, X_input, pipeline, missing_value, non_critical_feature, capsys):
        X_with_missing_value = X_input.copy()
        X_with_missing_value.loc[0, non_critical_feature] = missing_value  # use first row as a representative example
        expected_mode = X_with_missing_value[non_critical_feature].mode()[0]
        # Ensure .fit() prints warning message
        pipeline.fit(X_with_missing_value)
        captured_output_and_error = capsys.readouterr()
        warning_message = captured_output_and_error.out
        assert "Warning" in warning_message 
        assert "1 missing value found in non-critical features" in warning_message
        assert "will be imputed" in warning_message
        assert captured_output_and_error.err == ""
        # Ensure .fit() learns mode (most frequent value) of non-critical feature
        X_transformed = pipeline.transform(X_with_missing_value)
        assert X_transformed.loc[0, non_critical_feature] == expected_mode

    # Ensure pipeline .fit() raises MissingValueError for non-critical feature with only missing values
    @pytest.mark.integration
    @pytest.mark.parametrize("non_critical_feature", NON_CRITICAL_FEATURES)
    def test_missing_value_handling_pipeline_fit_raises_missing_value_error_for_non_critical_feature_with_only_missings(self, X_input, pipeline, non_critical_feature):
        # Create DataFrame with a non-critical feature with only missing values
        X_with_only_missings = X_input.copy()
        X_with_only_missings[non_critical_feature] = np.nan
        # Ensure .fit() raises MissingValueError
        expected_error_message = f"'{non_critical_feature}' cannot be only missing values. Please ensure at least one non-missing value."
        with pytest.raises(MissingValueError, match=expected_error_message):
            pipeline.fit(X_with_only_missings)

    # Ensure pipeline .fit() and .transform() raise ColumnMismatchError for missing columns in the input DataFrame
    @pytest.mark.integration
    @pytest.mark.parametrize("method", ["fit", "transform"])
    @pytest.mark.parametrize("missing_columns", [
        "income", 
        "age", 
        ["experience", "married"],
        ["house_ownership", "car_ownership"],
        ["profession", "city", "state"],
        ["current_job_yrs", "current_house_yrs"],
    ])
    def test_missing_value_handling_pipeline_raises_column_mismatch_error_for_missing_columns(self, X_input, pipeline, method, missing_columns):
        X = X_input.copy()
        X_with_missing_columns = X.drop(columns=missing_columns)
        # Ensure .fit() raises ColumnMismatchError 
        if method == "fit":
            with pytest.raises(ColumnMismatchError):
                pipeline.fit(X_with_missing_columns)
        # Ensure .transform() raises ColumnMismatchError 
        else:
            pipeline.fit(X)
            with pytest.raises(ColumnMismatchError):
                pipeline.transform(X_with_missing_columns)

    # Ensure pipeline .fit() and .transform() raise ColumnMismatchError for unexpected columns in the X input DataFrame
    @pytest.mark.integration
    @pytest.mark.parametrize("method", ["fit", "transform"])
    @pytest.mark.parametrize("unexpected_columns", [
        ["unexpected_column_1"],
        ["unexpected_column_1", "unexpected_column_2"],
        ["unexpected_column_1", "unexpected_column_2", "unexpected_column_3"]
    ])
    def test_missing_value_handling_pipeline_raises_column_mismatch_error_for_unexpected_columns(self, X_input, pipeline, method, unexpected_columns):
        X = X_input.copy()
        X_with_unexpected_columns = X_input.copy()
        for unexpected_column in unexpected_columns:
            X_with_unexpected_columns[unexpected_column] = 5 
        # Ensure .fit() raises ColumnMismatchError 
        if method == "fit":
            with pytest.raises(ColumnMismatchError):
                pipeline.fit(X_with_unexpected_columns)
        # Ensure .transform() raises ColumnMismatchError 
        else:
            pipeline.fit(X)
            with pytest.raises(ColumnMismatchError):
                pipeline.transform(X_with_unexpected_columns)

    # Ensure pipeline .transform() prints warning message and imputes mode for missing values in non-critical features
    @pytest.mark.integration
    @pytest.mark.parametrize("missing_value", [None, np.nan, pd.NA])
    @pytest.mark.parametrize("non_critical_feature", NON_CRITICAL_FEATURES)
    def test_missing_value_handling_pipeline_transform_warns_and_imputes_mode_for_missing_values_in_non_critical_features(self, X_input, pipeline, missing_value, non_critical_feature, capsys):
        X = X_input.copy()
        X_with_missing_value = X_input.copy()
        X_with_missing_value.loc[0, non_critical_feature] = missing_value  # use first row as a representative example
        expected_mode = X_with_missing_value[non_critical_feature].mode()[0]
        # Fit on original DataFrame, but transform on DataFrame with missing value
        pipeline.fit(X)
        X_transformed = pipeline.transform(X_with_missing_value)
        # Ensure .transform() prints warning message
        captured_output_and_error = capsys.readouterr()
        warning_message = captured_output_and_error.out
        assert "Warning" in warning_message 
        assert "1 missing value found in non-critical features" in warning_message
        assert "will be imputed" in warning_message
        assert captured_output_and_error.err == ""
        # Ensure .transform() imputes mode (most frequent value) 
        assert X_transformed.loc[0, non_critical_feature] == expected_mode
        # Ensure no more missing values on non-critical feature
        assert X_transformed[non_critical_feature].isna().sum() == 0    

    # Ensure pipeline .transform() on DataFrame with no missing values produces the expected column order 
    @pytest.mark.integration
    def test_missing_value_handling_pipeline_transform_with_no_missing_values_produces_expected_column_order(self, X_input, pipeline, capsys):
        X = X_input.copy()
        pipeline.fit(X)
        X_transformed = pipeline.transform(X)
        # Expected output (note: ColumnTransformer changes the column order)
        expected_column_order = NON_CRITICAL_FEATURES + CRITICAL_FEATURES
        expected_X_transformed = X[expected_column_order]
        # Ensure actual and expected output DataFrames are identical
        assert_frame_equal(X_transformed, expected_X_transformed)
        # Ensure no warnings in standard output and no errors
        captured_output_and_error = capsys.readouterr()
        assert captured_output_and_error.out == ""
        assert captured_output_and_error.err == ""
