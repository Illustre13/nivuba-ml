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
from src.pipeline import create_model_preprocessing_pipeline
from tests.integration.base_pipeline_tests import BasePipelineTests


# --- Fixtures ---
# Fixture to create X input DataFrame for use in tests
@pytest.fixture
def X_input():
    return pd.DataFrame({
        "income": [300000, 300000, 300000, 500000, 500000, 500000],
        "age": [30, 30, 30, 50, 50, 50],
        "experience": [3, 3, 3, 5, 5, 5],
        "married": [False, False, False, True, False, False],
        "house_ownership": ["rented", "rented", "norent_noown", "rented", "rented", "owned"],
        "car_ownership": [False, False, True, False, False, False],
        "profession": ["artist", "computer_hardware_engineer", "web_designer", "comedian", 
                        "financial_analyst", "statistician"],
        "city": ["sikar", "vellore", "bidar", "bongaigaon", "eluru[25]", "danapur"],
        "state": ["rajasthan", "tamil_nadu", "karnataka", "assam", "andhra_pradesh", "bihar"],
        "current_job_yrs": [3, 3, 3, 5, 5, 5],
        "current_house_yrs": [11, 11, 11, 13, 13, 13],
        "job_stability": ["variable", "moderate", "variable", "variable", "moderate", "moderate"],
        "city_tier": ["unknown", "unknown", "unknown", "unknown", "unknown", "tier_3"],
        "state_default_rate": [0.25, 0.25, 0.25, 0.75, 0.75, 0.75],
    })

# Fixture to create pipeline segment for use in tests
@pytest.fixture
def pipeline(): 
    return create_model_preprocessing_pipeline()


# --- TestModelPreprocessingPipeline class ---
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
class TestModelPreprocessingPipeline(BasePipelineTests):
    # Override parent class methods: ColumnTransformer accepts non-DataFrame inputs
    def test_pipeline_fit_and_transform_raise_type_error_if_X_not_df(self):
        pass  

    # Override parent class methods: ColumnTransformer accepts different column order compared to .fit()
    def test_pipeline_transform_raises_value_error_for_wrong_column_order(self):
        pass  
    
    # Ensure pipeline scales, encodes, and selects features correctly
    @pytest.mark.integration
    def test_model_preprocessing_pipeline_scales_encodes_and_selects_features(self, X_input, pipeline):
        X = X_input.copy() 
        pipeline.fit(X)
        X_transformed = pipeline.transform(X)
        expected_X_transformed = pd.DataFrame({
            "income": [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            "age": [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            "experience": [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            "current_job_yrs": [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            "current_house_yrs": [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            "state_default_rate": [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],            
            "house_ownership_owned": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "house_ownership_rented": [1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
            "job_stability": [0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            "city_tier": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            "married": [False, False, False, True, False, False],
            "car_ownership": [False, False, True, False, False, False],
        })
        assert_frame_equal(X_transformed, expected_X_transformed)

    # Ensure pipeline .fit() and .transform() raise ValueError for unknown categories (not specified in the "categories" parameter of OneHotEncoder or OrdinalEncoder) 
    @pytest.mark.integration
    @pytest.mark.parametrize("method", ["fit", "transform"])
    @pytest.mark.parametrize("categorical_column", ["house_ownership", "job_stability", "city_tier"])
    def test_model_preprocessing_pipeline_fit_and_transform_raise_value_error_for_unknown_categories(self, X_input, pipeline, method, categorical_column):
        X = X_input.copy()
        X_with_unknown_category = X_input.copy()
        X_with_unknown_category.loc[0, categorical_column] = "unknown_category"  # modify first row as a representative example
        if method == "fit":
            with pytest.raises(ValueError):
                pipeline.fit(X_with_unknown_category)
        else:  # method == "transform"
            pipeline.fit(X) 
            with pytest.raises(ValueError):
                pipeline.transform(X_with_unknown_category)

    # Ensure ColumnTransformer .transform() outputs the expected feature names
    @pytest.mark.integration
    def test_column_transformer_outputs_expected_feature_names(self, X_input, pipeline):
        X = X_input.copy()
        pipeline.fit(X)
        column_transformer = pipeline.named_steps["feature_scaler_encoder"]
        column_transformer_output = column_transformer.transform(X)
        expected_feature_names = [
            "income", "age", "experience", "current_job_yrs", "current_house_yrs", "state_default_rate", # from scaler
            "house_ownership_owned", "house_ownership_rented", # from nominal_encoder
            "job_stability", "city_tier", # from ordinal_encoder
            "married", "car_ownership", "profession", "city", "state" # from remainder="passthrough"            
        ]
        assert column_transformer_output.columns.tolist() == expected_feature_names

    # Ensure pipeline .fit() and .transform() raise ValueError for missing values
    @pytest.mark.integration
    @pytest.mark.parametrize("method", ["fit", "transform"])
    @pytest.mark.parametrize("column", ["house_ownership", "job_stability", "city_tier"])
    def test_model_preprocessing_pipeline_fit_and_transform_raise_value_error_for_missing_values(self, X_input, pipeline, method, column):
        X = X_input.copy()
        X_with_missing_value = X_input.copy()
        X_with_missing_value.loc[0, column] = np.nan
        if method == "fit":
            with pytest.raises(ValueError):
                pipeline.fit(X_with_missing_value)
        else:  # method == "transform"
            pipeline.fit(X) 
            with pytest.raises(ValueError):
                pipeline.transform(X_with_missing_value)
