# --- Imports ---
# Standard library imports
from unittest.mock import patch, MagicMock
from pathlib import Path

# Third-party library imports
import pytest
from fastapi.testclient import TestClient
from sklearn.pipeline import Pipeline
import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from pandas.testing import assert_frame_equal

# Local imports
from backend.app import load_pipeline, app


# --- Function .load_pipeline() ---
class TestLoadPipeline:
    @pytest.mark.unit
    @patch("backend.app.joblib.load")
    @patch("backend.app.Path.exists")
    def test_happy_path_with_mock_pipeline(self, mock_path_exists, mock_joblib_load):
        # Simulate that the file exists
        mock_path_exists.return_value = True
        # Simulate loaded pipeline instance
        mock_pipeline = MagicMock(spec=Pipeline)
        mock_pipeline.predict_proba = MagicMock()
        mock_joblib_load.return_value = mock_pipeline

        # Call .load_pipeline()
        pipeline = load_pipeline("some_path.joblib")

        # Ensure loaded pipeline is a mock pipeline with a "predict_proba" attribute
        assert pipeline is mock_pipeline
        assert hasattr(pipeline, "predict_proba")
        # Ensure Path.exists was called once
        mock_path_exists.assert_called_once()
        # Ensure joblib.load was called once
        mock_joblib_load.assert_called_once_with("some_path.joblib")

    @pytest.mark.unit
    @patch("backend.app.joblib.load")
    @patch("backend.app.Path.exists")
    def test_accepts_pathlike_object(self, mock_path_exists, mock_joblib_load):
        # Simulate that the file exists
        mock_path_exists.return_value = True
        # Simulate loaded pipeline instance
        mock_pipeline = MagicMock(spec=Pipeline)
        mock_pipeline.predict_proba = MagicMock()
        mock_joblib_load.return_value = mock_pipeline

        # Call .load_pipeline() with a pathlib.Path object (path-like object)
        path_like_object = Path("some_path.joblib")
        pipeline = load_pipeline(path_like_object)

        # Ensure loaded pipeline is a mock pipeline with a "predict_proba" attribute
        assert pipeline is mock_pipeline
        assert hasattr(pipeline, "predict_proba")
        # Ensure Path.exists was called once
        mock_path_exists.assert_called_once()
        # Ensure joblib.load was called once with converted string 
        mock_joblib_load.assert_called_once_with(str(path_like_object))

    @pytest.mark.unit
    @patch("backend.app.Path.exists")
    def test_raises_file_not_found_error_for_non_existent_file(self, mock_path_exists):
        # Simulate that the file does not exist
        mock_path_exists.return_value = False
        # Ensure FileNotFoundError is raised
        with pytest.raises(FileNotFoundError) as exc_info:
            load_pipeline("non_existent_file.joblib")
        # Ensure error message is as expected
        error_msg = str(exc_info.value)
        assert "Pipeline file not found at" in error_msg
        assert "non_existent_file.joblib" in error_msg
        # Ensure Path.exists was called 
        mock_path_exists.assert_called_once()

    @pytest.mark.unit
    @patch("backend.app.joblib.load")
    @patch("backend.app.Path.exists")
    def test_raises_runtime_error_if_joblib_load_fails(self, mock_path_exists, mock_joblib_load):
        # Simulate that the file exists
        mock_path_exists.return_value = True
        # Simulate an error when loading joblib file
        mock_joblib_load.side_effect = Exception("joblib load error")

        # Ensure RuntimeError is raised
        with pytest.raises(RuntimeError) as exc_info:
            load_pipeline("some_path.joblib")
        # Ensure error message is as expected
        error_msg = str(exc_info.value)
        assert "Failed to load pipeline" in error_msg
        assert "some_path.joblib" in error_msg
        # Ensure the original error was propagated
        propagated_error = exc_info.value.__cause__
        assert isinstance(propagated_error, Exception)
        assert str(propagated_error) == "joblib load error"
        # Ensure Path.exists() and joblib.load() were called
        mock_path_exists.assert_called_once()
        mock_joblib_load.assert_called_once_with("some_path.joblib")

    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_type", [
        "a string",
        1,
        1.23,
        False,
        None,  
        ["a", "list"],
        ("a", "tuple"),
        {"a": "dictionary"},
        {"a", "set"}
    ])
    @patch("backend.app.joblib.load")
    @patch("backend.app.Path.exists")
    def test_raises_type_error_if_loaded_object_is_not_pipeline(self, mock_path_exists, mock_joblib_load, invalid_type):
        # Simulate that pipeline file exists
        mock_path_exists.return_value = True
        # Simulate loaded object that is not a pipeline
        mock_joblib_load.return_value = invalid_type

        # Ensure .load_pipeline() raises TypeError
        with pytest.raises(TypeError) as exc_info:
            load_pipeline("some_path.joblib")
        # Ensure error message is as expected
        error_msg = str(exc_info.value)
        assert "Loaded object is not a scikit-learn Pipeline" in error_msg
        # Ensure Path.exists() and joblib.load() were called
        mock_path_exists.assert_called_once()
        mock_joblib_load.assert_called_once_with("some_path.joblib")

    @pytest.mark.unit
    @patch("backend.app.joblib.load")
    @patch("backend.app.Path.exists")
    def test_raises_type_error_if_predict_proba_does_not_exist(self, mock_path_exists, mock_joblib_load):
        # Simulate that pipeline file exists
        mock_path_exists.return_value = True 
        # Simulate loaded pipeline instance without a "predict_proba" method
        mock_pipeline = MagicMock(spec=Pipeline)
        del mock_pipeline.predict_proba
        mock_joblib_load.return_value = mock_pipeline

        # Ensure .load_pipeline() raises TypeError
        with pytest.raises(TypeError) as exc_info:
            load_pipeline("some_path.joblib")
        # Ensure error message is as expected
        error_msg = str(exc_info.value)
        assert "Loaded pipeline does not have a .predict_proba() method" in error_msg
        # Ensure Path.exists() and joblib.load() were called
        mock_path_exists.assert_called_once()
        mock_joblib_load.assert_called_once_with("some_path.joblib")


# --- FastAPI endpoint "/predict" ---
client = TestClient(app)

class TestPredict:
    @pytest.mark.unit
    @patch("backend.app.pipeline.predict_proba")
    def test_happy_path_single_input(self, mock_predict_proba):
        valid_single_input = {
            "income": 300000,
            "age": 30,
            "experience": 3,
            "married": "single",
            "house_ownership": "rented",
            "car_ownership": "no",
            "profession": "artist",
            "city": "sikar",
            "state": "rajasthan",
            "current_job_yrs": 3,
            "current_house_yrs": 11           
        }
        # Simulate the return value of pipeline.predict_proba()
        mock_predict_proba.return_value = np.array([[0.8, 0.2]])
        
        # Make post request to predict endpoint with test client
        response = client.post("/predict", json=valid_single_input)

        # Ensure post request was successful
        assert response.status_code == 200
        # Ensure prediction response is as expected
        prediction_response = response.json()
        expected_prediction_response = {
            "results": [{
                "prediction": "No Default",  # 0.2 < 0.29 threshold
                "probabilities": {
                    "Default": 0.2, 
                    "No Default": 0.8
                }
            }],
            "n_predictions": 1
        }
        assert prediction_response == expected_prediction_response

    @pytest.mark.unit
    @patch("backend.app.pipeline.predict_proba")
    def test_happy_path_batch_input(self, mock_predict_proba):
        valid_batch_input = [
            {
                "income": 300000,
                "age": 30,
                "experience": 3,
                "married": "single",
                "house_ownership": "rented",
                "car_ownership": "no",
                "profession": "artist",
                "city": "sikar",
                "state": "rajasthan",
                "current_job_yrs": 3,
                "current_house_yrs": 11           
            },
            {
                "income": 1000000,
                "age": 30,
                "experience": 10,
                "married": "married",
                "house_ownership": "rented",
                "car_ownership": "yes",
                "profession": "architect",
                "city": "delhi_city",
                "state": "assam",
                "current_job_yrs": 7,
                "current_house_yrs": 12           
            }            
        ]
        # Simulate the return value of pipeline.predict_proba()
        mock_predict_proba.return_value = np.array([[0.8, 0.2], [0.3, 0.7]])
        
        # Make post request to predict endpoint 
        response = client.post("/predict", json=valid_batch_input)

        # Ensure post request was successful
        assert response.status_code == 200
        # Ensure prediction response is as expected
        prediction_response = response.json()
        expected_prediction_response = {
            "results": [
                {
                    "prediction": "No Default",  # 0.2 < 0.29 threshold
                    "probabilities": {
                        "Default": 0.2, 
                        "No Default": 0.8
                    }
                },
                {
                    "prediction": "Default",  
                    "probabilities": {
                        "Default": 0.7, 
                        "No Default": 0.3
                    }
                }
            ],
            "n_predictions": 2
        }
        assert prediction_response == expected_prediction_response

    @pytest.mark.unit
    @patch("backend.app.pipeline.predict_proba")
    def test_standardize_input_happy_path(self, mock_predict_proba):
        valid_single_input = {
            "income": 300000,
            "age": 30,
            "experience": 3,
            "married": "single",
            "house_ownership": "rented",
            "car_ownership": "no",
            "profession": "artist",
            "city": "sikar",
            "state": "rajasthan",
            "current_job_yrs": 3,
            "current_house_yrs": 11           
        }
        # Simulate the return value of pipeline.predict_proba()
        mock_predict_proba.return_value = np.array([[0.8, 0.2]])
        
        # Make post request to predict endpoint 
        client.post("/predict", json=valid_single_input)

        # Ensure .predict_proba() was called once
        mock_predict_proba.assert_called_once()
        # Get positional and keyword arguments used in the .predict_proba() method call
        args, kwargs = mock_predict_proba.call_args
        # Get DataFrame used in call (first positional argument)
        df = args[0]
        # Ensure .predict_proba() was called with the expected DataFrame
        expected_df = pd.DataFrame([valid_single_input])
        assert_frame_equal(df, expected_df)

    @pytest.mark.unit
    @pytest.mark.parametrize("predicted_probabilities, expected_predictions", [
        (np.array([[0.8, 0.2]]), np.array([False])), 
        (np.array([[0.2, 0.8]]), np.array([True])),  
        (np.array([[0.71, 0.29]]), np.array([True])),  # threshold value
    ])
    @patch("backend.app.pipeline.predict_proba")
    @patch("backend.app.zip")
    def test_apply_threshold_happy_path(self, mock_zip, mock_predict_proba, predicted_probabilities, expected_predictions):
        valid_single_input = {
            "income": 300000,
            "age": 30,
            "experience": 3,
            "married": "single",
            "house_ownership": "rented",
            "car_ownership": "no",
            "profession": "artist",
            "city": "sikar",
            "state": "rajasthan",
            "current_job_yrs": 3,
            "current_house_yrs": 11           
        }
        # Simulate the return value of pipeline.predict_proba()
        mock_predict_proba.return_value = predicted_probabilities

        # Make post request to predict endpoint 
        client.post("/predict", json=valid_single_input)

        # Ensure .predict_proba() and .zip() were called once
        mock_predict_proba.assert_called_once()
        mock_zip.assert_called_once()
        # Ensure .zip() was called with the expected predictions
        predictions = mock_zip.call_args[0][0]
        assert_array_equal(predictions, expected_predictions)

    @pytest.mark.unit
    @patch("backend.app.pipeline.predict_proba")
    @patch("backend.app.zip")
    def test_create_prediction_response_happy_path(self, mock_zip, mock_predict_proba):
        valid_single_input = {
            "income": 300000,
            "age": 30,
            "experience": 3,
            "married": "single",
            "house_ownership": "rented",
            "car_ownership": "no",
            "profession": "artist",
            "city": "sikar",
            "state": "rajasthan",
            "current_job_yrs": 3,
            "current_house_yrs": 11           
        }
        # Simulate the return value of pipeline.predict_proba()
        mock_predict_proba.return_value = np.array([[0.8, 0.2]])
        # Simulate the return value of zip(predictions, predicted_probabilities)
        mock_zip.return_value = [
            (False, np.array([0.8, 0.2]))
        ]

        # Make post request to predict endpoint 
        response = client.post("/predict", json=valid_single_input)

        # Ensure .predict_proba() and .zip() were called once
        mock_predict_proba.assert_called_once()
        mock_zip.assert_called_once()
        # Ensure prediction response is as expected
        prediction_response = response.json()
        expected_prediction_response = {
            "results": [{
                "prediction": "No Default",  
                "probabilities": {
                    "Default": 0.2, 
                    "No Default": 0.8
                }
            }],
            "n_predictions": 1
        }
        assert prediction_response == expected_prediction_response

    # Pipeline failure 
    @pytest.mark.unit
    def test_return_http_500_for_pipeline_failure(self, monkeypatch):
        valid_single_input = {
            "income": 300000,
            "age": 30,
            "experience": 3,
            "married": "single",
            "house_ownership": "rented",
            "car_ownership": "no",
            "profession": "artist",
            "city": "sikar",
            "state": "rajasthan",
            "current_job_yrs": 3,
            "current_house_yrs": 11           
        }
        # Use pytest's built-in monkeypatch fixture with a custom function to simulate failure when calling .predict_proba()
        def simulate_predict_proba_failure(X: pd.DataFrame):
            raise RuntimeError("Simulated failure of .predict_proba()")
        monkeypatch.setattr("backend.app.pipeline.predict_proba", simulate_predict_proba_failure)

        # Post request to predict endpoint 
        response = client.post("/predict", json=valid_single_input)

        # Ensure response status code is 500
        assert response.status_code == 500
        # Ensure error message is as expected
        assert "Internal server error during loan default prediction" in response.text 

    # Prediction response creation failure 
    @pytest.mark.unit
    @patch("backend.app.pipeline.predict_proba")
    def test_return_http_500_for_malformed_pipeline_output(self, mock_predict_proba):
        valid_single_input = {
            "income": 300000,
            "age": 30,
            "experience": 3,
            "married": "single",
            "house_ownership": "rented",
            "car_ownership": "no",
            "profession": "artist",
            "city": "sikar",
            "state": "rajasthan",
            "current_job_yrs": 3,
            "current_house_yrs": 11           
        }
        # Simulate malformed pipeline output of .predict_proba()
        mock_predict_proba.return_value = np.array([[0.8]])  # instead of np.array([[0.8, 0.2]]) 

        # Post request to predict endpoint 
        response = client.post("/predict", json=valid_single_input)

        # Ensure .predict_proba() was called once
        mock_predict_proba.assert_called_once()
        # Ensure response status code is 500 (Internal Server Error)
        assert response.status_code == 500
        # Ensure error message is as expected
        assert "Internal server error during loan default prediction" in response.text 
