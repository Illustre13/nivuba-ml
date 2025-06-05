import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
import pandas as pd
from pandas.testing import assert_frame_equal, assert_index_equal
import numpy as np
import pickle


# Base tests for a sklearn pipeline or subsegment of a pipeline (that individual integration test classes can inherit from)
class BasePipelineTests:
    # Ensure pipeline instance can be cloned 
    @pytest.mark.integration
    def test_pipeline_can_be_cloned(self, pipeline, X_input):
        X = X_input.copy()
        cloned_pipeline = clone(pipeline)
        pipeline.fit(X)
        pipeline_output = pipeline.transform(X)
        cloned_pipeline.fit(X)
        cloned_pipeline_output = cloned_pipeline.transform(X)
        # Ensure it's a new object, not a pointer to the old one
        assert cloned_pipeline is not pipeline
        # Ensure the original and cloned outputs are identical
        assert_frame_equal(pipeline_output, cloned_pipeline_output)

    # Ensure equal output of .fit().transform() and .fit_transform()
    @pytest.mark.integration
    def test_pipeline_fit_transform_equivalence(self, pipeline, X_input):
        X = X_input.copy()
        # Create two transformer instances
        pipeline_1 = clone(pipeline)
        pipeline_2 = clone(pipeline)
        # Ensure they are different objects in memory
        assert pipeline_1 is not pipeline_2
        # Perform .fit().transform() vs .fit_transform()
        X_fit_then_transform = pipeline_1.fit(X).transform(X) 
        X_fit_transform = pipeline_2.fit_transform(X)
        # Ensure the output DataFrames are identical
        assert_frame_equal(X_fit_then_transform, X_fit_transform)

    # Ensure pipeline .fit() and .transform() raise TypeError if "X" input is not a pandas DataFrame
    @pytest.mark.integration
    @pytest.mark.parametrize("method", ["fit", "transform"])
    @pytest.mark.parametrize("invalid_X_input", [
        np.array([[1, 2], [3, 4]]), 
        pd.Series([1, 2, 3]),
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
    def test_pipeline_fit_and_transform_raise_type_error_if_X_not_df(self, pipeline, X_input, method, invalid_X_input):
        expected_error_message = "Input X must be a pandas DataFrame."
        # Ensure .fit() raises TypeError
        if method == "fit":
            with pytest.raises(TypeError, match=expected_error_message):
                pipeline.fit(invalid_X_input)
        # Ensure .transform() raises TypeError
        else:
            X = X_input.copy()
            pipeline.fit(X)
            with pytest.raises(TypeError, match=expected_error_message):
                pipeline.transform(invalid_X_input)

    # Ensure pipeline .transform() raises NotFittedError if pipeline instance has not been fitted yet
    @pytest.mark.integration
    def test_pipeline_transform_raises_not_fitted_error_if_unfitted(self, pipeline, X_input):
        X = X_input.copy()
        # .fit() is intentionally not called here
        with pytest.raises(NotFittedError):
            pipeline.transform(X)

    # Ensure pipeline .transform() does not modify the "X" input DataFrame
    @pytest.mark.integration
    def test_pipeline_transform_does_not_modify_input_df(self, pipeline, X_input):
        X_original = X_input.copy()
        pipeline.fit(X_original)
        pipeline.transform(X_original)
        assert_frame_equal(X_original, X_input)

    # Ensure fitted pipeline instance can be pickled and unpickled without losing its attributes and functionality
    @pytest.mark.integration
    def test_fitted_pipeline_can_be_pickled(self, pipeline, X_input):
        X = X_input.copy()
        pipeline.fit(X)
        X_transformed = pipeline.transform(X)
        # Pickle and unpickle
        pickled_pipeline = pickle.dumps(pipeline)
        unpickled_pipeline = pickle.loads(pickled_pipeline)
        # Ensure that unpickled pipeline produces identical output as original
        unpickled_X_transformed = unpickled_pipeline.transform(X)
        assert_frame_equal(unpickled_X_transformed, X_transformed)

    # Ensure pipeline .transform() raises ValueError if columns are in different order than during .fit()
    @pytest.mark.integration
    def test_pipeline_transform_raises_value_error_for_wrong_column_order(self, pipeline, X_input):
        X = X_input.copy()
        # Fit on original DataFrame X
        pipeline.fit(X)
        # Create DataFrame with different column order than during .fit()
        reversed_columns = X.columns[::-1]  # reverse order as an example
        X_with_wrong_column_order = X[reversed_columns]  
        # Ensure .transform() on wrong column order raises ValueError
        expected_error_message = "Feature names and feature order of input X must be the same as during .fit()."
        with pytest.raises(ValueError, match=expected_error_message):
            pipeline.transform(X_with_wrong_column_order)

    # Ensure pipeline .transform() preserves the index of the input DataFrame
    @pytest.mark.integration
    def test_pipeline_transform_preserves_df_index(self, pipeline, X_input):
        X = X_input.copy()
        pipeline.fit(X)
        X_transformed = pipeline.transform(X)
        # Ensure that indexes of transformed and original DataFrame are identical
        assert_index_equal(X_transformed.index, X.index)

    # Ensure pipeline .transform() passes through an empty DataFrame
    @pytest.mark.unit
    def test_pipeline_transform_passes_through_empty_df(self, pipeline, X_input):
        X = X_input.copy()
        # Create empty DataFrame
        input_columns = X.columns
        input_data_types = X.dtypes.to_dict()
        X_empty = pd.DataFrame(columns=input_columns).astype(input_data_types)
        # Fit and transform on non-empty DataFrame
        pipeline.fit(X)
        X_transformed = pipeline.transform(X)
        # Transform on empty Dataframe     
        X_transformed_empty = pipeline.transform(X_empty)
        # Create expected output (based on non-empty DataFrame)
        expected_output_columns = X_transformed.columns
        expected_output_data_types = X_transformed.dtypes.to_dict()
        expected_X_transformed_empty = pd.DataFrame(columns=expected_output_columns).astype(expected_output_data_types)
        # Ensure actual and expected output DataFrames are identical
        assert_frame_equal(X_transformed_empty, expected_X_transformed_empty)


# Base tests for a pipeline with supervised transformers (inherits from BasePipelineTests)
# Overrides methods that call .fit() to include the target variable "y"
class BaseSupervisedPipelineTests(BasePipelineTests):
    # Ensure pipeline instance can be cloned 
    @pytest.mark.integration
    def test_pipeline_can_be_cloned(self, pipeline, X_input, y_input):
        X = X_input.copy()
        y = y_input.copy()
        cloned_pipeline = clone(pipeline)
        pipeline.fit(X, y)
        pipeline_output = pipeline.transform(X)
        cloned_pipeline.fit(X, y)
        cloned_pipeline_output = cloned_pipeline.transform(X)
        # Ensure it's a new object, not a pointer to the old one
        assert cloned_pipeline is not pipeline
        # Ensure the original and cloned outputs are identical
        assert_frame_equal(pipeline_output, cloned_pipeline_output)

    # Ensure equal output of .fit().transform() and .fit_transform()
    @pytest.mark.integration
    def test_pipeline_fit_transform_equivalence(self, pipeline, X_input, y_input):
        X = X_input.copy()
        y = y_input.copy()
        # Create two transformer instances
        pipeline_1 = clone(pipeline)
        pipeline_2 = clone(pipeline)
        # Ensure they are different objects in memory
        assert pipeline_1 is not pipeline_2
        # Perform .fit().transform() vs .fit_transform()
        X_fit_then_transform = pipeline_1.fit(X, y).transform(X) 
        X_fit_transform = pipeline_2.fit_transform(X, y)
        # Ensure the output DataFrames are identical
        assert_frame_equal(X_fit_then_transform, X_fit_transform)

    # Ensure pipeline .fit() and .transform() raise TypeError if "X" input is not a pandas DataFrame
    @pytest.mark.integration
    @pytest.mark.parametrize("method", ["fit", "transform"])
    @pytest.mark.parametrize("invalid_X_input", [
        np.array([[1, 2], [3, 4]]), 
        pd.Series([1, 2, 3]),
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
    def test_pipeline_fit_and_transform_raise_type_error_if_X_not_df(self, pipeline, X_input, y_input, method, invalid_X_input):
        y = y_input.copy()
        expected_error_message = "Input X must be a pandas DataFrame."
        # Ensure .fit() raises TypeError
        if method == "fit":
            with pytest.raises(TypeError, match=expected_error_message):
                pipeline.fit(invalid_X_input, y)
        # Ensure .transform() raises TypeError
        else:
            X = X_input.copy()
            pipeline.fit(X, y)
            with pytest.raises(TypeError, match=expected_error_message):
                pipeline.transform(invalid_X_input)

    # Ensure pipeline .transform() does not modify the "X" input DataFrame
    @pytest.mark.integration
    def test_pipeline_transform_does_not_modify_input_df(self, pipeline, X_input, y_input):
        X_original = X_input.copy()
        y = y_input.copy()
        pipeline.fit(X_original, y)
        pipeline.transform(X_original)
        assert_frame_equal(X_original, X_input)

    # Ensure fitted pipeline instance can be pickled and unpickled without losing its attributes and functionality
    @pytest.mark.integration
    def test_fitted_pipeline_can_be_pickled(self, pipeline, X_input, y_input):
        X = X_input.copy()
        y = y_input.copy()
        pipeline.fit(X, y)
        X_transformed = pipeline.transform(X)
        # Pickle and unpickle
        pickled_pipeline = pickle.dumps(pipeline)
        unpickled_pipeline = pickle.loads(pickled_pipeline)
        # Ensure that unpickled pipeline produces identical output as original
        unpickled_X_transformed = unpickled_pipeline.transform(X)
        assert_frame_equal(unpickled_X_transformed, X_transformed)

    # Ensure pipeline .transform() raises ValueError if columns are in different order than during .fit()
    @pytest.mark.integration
    def test_pipeline_transform_raises_value_error_for_wrong_column_order(self, pipeline, X_input, y_input):
        X = X_input.copy()
        y = y_input.copy()
        # Fit on original DataFrame X
        pipeline.fit(X, y)
        # Create DataFrame with different column order than during .fit()
        reversed_columns = X.columns[::-1]  # reverse order as an example
        X_with_wrong_column_order = X[reversed_columns]  
        # Ensure .transform() on wrong column order raises ValueError
        expected_error_message = "Feature names and feature order of input X must be the same as during .fit()."
        with pytest.raises(ValueError, match=expected_error_message):
            pipeline.transform(X_with_wrong_column_order)

    # Ensure pipeline .transform() preserves the index of the input DataFrame
    @pytest.mark.integration
    def test_pipeline_transform_preserves_df_index(self, pipeline, X_input, y_input):
        X = X_input.copy()
        y = y_input.copy()
        pipeline.fit(X, y)
        X_transformed = pipeline.transform(X)
        # Ensure that indexes of transformed and original DataFrame are identical
        assert_index_equal(X_transformed.index, X.index)

    # Ensure pipeline .transform() passes through an empty DataFrame
    @pytest.mark.unit
    def test_pipeline_transform_passes_through_empty_df(self, pipeline, X_input, y_input):
        X = X_input.copy()
        y = y_input.copy()
        # Create empty DataFrame
        input_columns = X.columns
        input_data_types = X.dtypes.to_dict()
        X_empty = pd.DataFrame(columns=input_columns).astype(input_data_types)
        # Fit and transform on non-empty DataFrame
        pipeline.fit(X, y)
        X_transformed = pipeline.transform(X)
        # Transform on empty Dataframe     
        X_transformed_empty = pipeline.transform(X_empty)
        # Create expected output (based on non-empty DataFrame)
        expected_output_columns = X_transformed.columns
        expected_output_data_types = X_transformed.dtypes.to_dict()
        expected_X_transformed_empty = pd.DataFrame(columns=expected_output_columns).astype(expected_output_data_types)
        # Ensure actual and expected output DataFrames are identical
        assert_frame_equal(X_transformed_empty, expected_X_transformed_empty)
