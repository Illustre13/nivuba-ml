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
from src.custom_transformers import MissingValueStandardizer
from tests.unit.base_transformer_tests import BaseTransformerTests


# --- Fixtures ---
# Fixture to instantiate MissingValueStandardizer class for use in tests
@pytest.fixture
def transformer():
    return MissingValueStandardizer()

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


# --- TestMissingValueStandardizer class ---
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
class TestMissingValueStandardizer(BaseTransformerTests):
    # Class instantiation 
    @pytest.mark.unit
    def test_instantiation(self, transformer):
        # First, run the .test_instantiation() method from the parent class BaseTransformerTests
        super().test_instantiation(transformer)
        # Then, add assertions specific to the MissingValueChecker class
        assert isinstance(transformer, MissingValueStandardizer)

    # Overwrite parent class method because this transformer allows extra columns
    @pytest.mark.unit
    def test_transform_raises_value_error_for_extra_column(self):
        pass


    # Overwrite parent class method because this transformer allows different column order compared to .fit()
    @pytest.mark.unit
    def test_transform_raises_value_error_for_wrong_column_order(self):
        pass

    # Ensure .transform() standardizes all missing value types to np.nan
    @pytest.mark.unit
    @pytest.mark.parametrize("column", ["income", "city"])  # test both numeric "income" and object type "city"
    @pytest.mark.parametrize("missing_value", [None, np.nan, pd.NA])
    def test_transform_standardizes_all_missing_value_types_to_numpy_nan(self, transformer, X_input, column, missing_value):
        X = X_input.copy()
        transformer.fit(X)
        X_with_missing_value = X_input.copy()
        X_with_missing_value.loc[0, column] = missing_value  # use first row of first column as a representative example
        X_transformed = transformer.transform(X_with_missing_value)
        assert np.isnan(X_transformed.loc[0, column])
