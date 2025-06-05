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
from src.pipeline import create_feature_engineering_pipeline
from src.custom_transformers import ColumnMismatchError, CategoricalLabelError, MissingValueError
from tests.integration.base_pipeline_tests import BaseSupervisedPipelineTests


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

# Fixture to create y input Series for use in tests
@pytest.fixture
def y_input():
    return pd.Series([0, 1, 0, 0, 1, 0])

# Fixture to create pipeline segment for use in tests
@pytest.fixture
def pipeline(): 
    return create_feature_engineering_pipeline()


# --- TestFeatureEngineeringPipeline class ---
# Inherits from BaseSupervisedPipelineTests which adds the following integration tests:
# .test_pipeline_can_be_cloned()
# .test_pipeline_fit_transform_equivalence()
# .test_pipeline_fit_and_transform_raise_type_error_if_X_not_df()
# .test_pipeline_transform_does_not_modify_input_df()
# .test_fitted_pipeline_can_be_pickled()
# .test_pipeline_transform_raises_value_error_for_wrong_column_order()
# .test_pipeline_transform_preserves_df_index()
# .test_pipeline_transform_passes_through_empty_df()
# BaseSupervisedPipelineTests further inherits the following test from BasePipelineTests:
# .test_pipeline_transform_raises_not_fitted_error_if_unfitted()
class TestFeatureEngineeringPipeline(BaseSupervisedPipelineTests):
    # Ensure pipeline correctly engineers features
    @pytest.mark.integration
    def test_feature_engineering_pipeline_happy_path(self, pipeline, X_input, y_input):
        X = X_input.copy()
        y = y_input.copy()
        pipeline.fit(X, y)
        X_transformed = pipeline.transform(X)
        expected_X_transformed = pd.DataFrame({
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
            "state_default_rate": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        })
        # Ensure actual and expected output DataFrames are identical
        assert_frame_equal(X_transformed, expected_X_transformed)

    # Ensure pipeline .fit() and .transform() raise ColumnMismatchError for missing columns in the input DataFrame
    @pytest.mark.integration
    @pytest.mark.parametrize("method", ["fit", "transform"])
    @pytest.mark.parametrize("missing_column", [
        "married", 
        "car_ownership", 
        "profession", 
        "city", 
        "state"
    ])
    def test_feature_engineering_pipeline_fit_and_transform_raise_column_mismatch_error_for_missing_columns(self, X_input, y_input, pipeline, method, missing_column):
        X = X_input.copy()
        y = y_input.copy()
        X_with_missing_column = X.drop(columns=missing_column)
        # Ensure .fit() raises ColumnMismatchError 
        if method == "fit":
            with pytest.raises(ColumnMismatchError):
                pipeline.fit(X_with_missing_column, y)
        # Ensure .transform() raises ColumnMismatchError 
        else:
            pipeline.fit(X, y)
            with pytest.raises(ColumnMismatchError):
                pipeline.transform(X_with_missing_column)

    # Ensure pipeline .fit() and .transform() raise CategoricalLabelError for unknown labels 
    @pytest.mark.integration
    @pytest.mark.parametrize("method", ["fit", "transform"])
    @pytest.mark.parametrize("column", ["married", "car_ownership", "profession", "city"])
    def test_feature_engineering_pipeline_fit_and_transform_raise_categorical_label_error_for_unknown_labels(self, X_input, y_input, pipeline, method, column):
        X = X_input.copy()
        y = y_input.copy()
        X_with_unknown_label = X_input.copy()
        X_with_unknown_label.loc[0, column] = "unknown_label"  # modify first row as a representative example
        if method == "fit":
            with pytest.raises(CategoricalLabelError):
                pipeline.fit(X_with_unknown_label, y)
        else:  # method == "transform"
            pipeline.fit(X, y) 
            with pytest.raises(CategoricalLabelError):
                pipeline.transform(X_with_unknown_label)

    # Ensure pipeline .transform() raises CategoricalLabelError for states not seen during .fit() 
    @pytest.mark.integration
    def test_feature_engineering_pipeline_transform_raises_categorical_label_error_for_unknown_states(self, X_input, y_input, pipeline):
        X = X_input.copy()
        y = y_input.copy()
        X_with_unknown_state = X_input.copy()
        X_with_unknown_state.loc[0, "state"] = "unknown_state"  # modify first row as a representative example
        pipeline.fit(X, y) 
        with pytest.raises(CategoricalLabelError):
            pipeline.transform(X_with_unknown_state)

    # Ensure pipeline .fit() and .transform() raise MissingValueError for missing values on required columns 
    @pytest.mark.integration
    @pytest.mark.parametrize("method", ["fit", "transform"])
    @pytest.mark.parametrize("required_column, expected_error_message", [
        ("married", "'married' column cannot contain missing values"), 
        ("car_ownership", "'car_ownership' column cannot contain missing values"), 
        ("profession", "'profession' column cannot contain missing values"), 
        ("city", "'city' column cannot contain missing values"),
        ("state", "'state' column cannot contain missing values")
    ])
    def test_feature_engineering_pipeline_fit_and_transform_raise_missing_value_error_for_required_columns(self, X_input, y_input, pipeline, method, required_column, expected_error_message):
        X = X_input.copy()
        y = y_input.copy()
        X_with_missing_value = X_input.copy()
        X_with_missing_value.loc[0, required_column] = np.nan  # modify first row as a representative example
        if method == "fit":
            with pytest.raises(MissingValueError, match=expected_error_message):
                pipeline.fit(X_with_missing_value, y)
        else:  # method == "transform"
            pipeline.fit(X, y)
            with pytest.raises(MissingValueError, match=expected_error_message):
                pipeline.transform(X_with_missing_value)
