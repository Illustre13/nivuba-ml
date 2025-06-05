# --- Imports ---
# Standard library imports
from pathlib import Path 

# Third-party library imports
import pytest 
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import joblib
from fastapi.testclient import TestClient

# Local imports
from backend.app import load_pipeline, app


# --- Function .load_pipeline() ---
class TestLoadPipeline:
    @pytest.mark.integration
    @pytest.mark.parametrize("path_input_type", ["str", "Path"])
    def test_happy_path_with_minimal_pipeline(self, path_input_type, tmp_path):
        # Create minimal pipeline
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("rf_classifier", RandomForestClassifier())
        ])
        # Train with minimal data
        X = [[0, 0], [1, 1]]
        y = [0, 1]
        pipeline.fit(X, y)
        # Create path to pipeline as both Path object and string
        pipeline_path: Path = tmp_path / "example_pipeline.joblib"
        pipeline_path_str: str = str(pipeline_path)

        # Save to temporary file
        joblib.dump(pipeline, pipeline_path_str)

        # Call .load_pipeline() function
        if path_input_type == "str":
            loaded_pipeline = load_pipeline(pipeline_path_str)
        else:  # path_input_type == "Path"
            loaded_pipeline = load_pipeline(pipeline_path)

        # Call .pred_proba() method (with single sample)
        predicted_probabilities = loaded_pipeline.predict_proba([[0, 0]])

        # Ensure pipeline is scikit-learn Pipeline class with "predict_proba" method
        assert isinstance(loaded_pipeline, Pipeline)
        assert hasattr(loaded_pipeline, "predict_proba")
        # Ensure .predict_proba() returns numpy 2darray with 1 row and two columns (for classes 0 and 1)
        assert predicted_probabilities.shape == (1, 2)

    @pytest.mark.integration
    @pytest.mark.parametrize("path_input_type", ["str", "Path"])
    def test_happy_path_with_real_pipeline(self, path_input_type, request):
        # Get LocalPath to root directory using pytest's built-in request fixture
        root_dir = request.config.rootdir
        # Convert LocalPath to Path object 
        root_dir = Path(str(root_dir))
        # Get path to pipeline 
        pipeline_path = root_dir / "models" / "loan_default_rf_pipeline.joblib"  
        # Valid pipeline input (single-row DataFrame)
        valid_input = pd.DataFrame({
            "income": [300000],
            "age": [30],
            "experience": [3],
            "married": ["single"],
            "house_ownership": ["rented"],
            "car_ownership": ["no"],
            "profession": ["Artist"],
            "city": ["Sikar"],
            "state": ["Rajasthan"],
            "current_job_yrs": [3],
            "current_house_yrs": [11],
        })  

        # Call .load_pipeline() function
        if path_input_type == "str":
            pipeline_path = str(pipeline_path)
        loaded_pipeline = load_pipeline(pipeline_path)
        # Predict probabilities with valid input
        predicted_probabilities = loaded_pipeline.predict_proba(valid_input)

        # Ensure pipeline is scikit-learn Pipeline class with "predict_proba" method
        assert isinstance(loaded_pipeline, Pipeline)
        assert hasattr(loaded_pipeline, "predict_proba")
        # Ensure .pred_proba() returns numpy array with 1 row and 2 columns (for the 2 classes)
        assert isinstance(predicted_probabilities, np.ndarray)
        assert predicted_probabilities.shape == (1, 2)

    @pytest.mark.integration
    @pytest.mark.parametrize("invalid_path_type", [
        None,  
        1,
        1.23,
        False,
        ["a", "list"],
        ("a", "tuple"),
        {"a": "dictionary"},
        {"a", "set"}
    ])
    def test_raises_type_error_for_invalid_path_type(self, invalid_path_type):
        # Ensure .load_pipeline() raises TypeError
        with pytest.raises(TypeError) as exc_info:
            load_pipeline(invalid_path_type)
        # Ensure error message is as expected
        error_msg = str(exc_info.value)
        assert "Error when loading pipeline"
        assert "'path' must be a string or Path object" in error_msg   
        assert f"{type(invalid_path_type).__name__}" in error_msg

    @pytest.mark.integration
    def test_raises_file_not_found_error_for_non_existent_file(self):
        # Ensure .load_pipeline() raises FileNotFoundError
        with pytest.raises(FileNotFoundError) as exc_info:
            load_pipeline("non_existent_file.joblib")
        # Ensure error message is as expected
        error_msg = str(exc_info.value)
        assert "Error when loading pipeline"
        assert "File not found at" in error_msg
        assert "non_existent_file.joblib" in error_msg

    @pytest.mark.integration
    def test_raises_runtime_error_if_joblib_load_fails(self, tmp_path):
        # Create a corrupt (non-joblib) file
        corrupt_file_path = tmp_path / "corrupt.joblib"
        with open(corrupt_file_path, "w") as file:
            file.write("This is not a joblib file")

        # Ensure .load_pipeline() raises RuntimeError 
        with pytest.raises(RuntimeError) as exc_info:
            load_pipeline(corrupt_file_path)
        # Ensure error message is as expected
        error_msg = str(exc_info.value)
        assert "Error when loading pipeline" in error_msg
        assert "corrupt.joblib" in error_msg

    @pytest.mark.integration
    def test_raises_type_error_if_loaded_object_is_not_pipeline(self, tmp_path):
        # Create a joblib file that is not a Pipeline
        not_a_pipeline = {"a": "dictionary"}
        file_path = tmp_path / "not_a_pipeline.joblib"
        joblib.dump(not_a_pipeline, file_path)

        # Ensure .load_pipeline() raises TypeError
        with pytest.raises(TypeError) as exc_info:
            load_pipeline(file_path)
        # Ensure error message is as expected
        error_msg = str(exc_info.value)
        assert "Error when loading pipeline"
        assert "Loaded object is not a scikit-learn Pipeline" in error_msg    

    @pytest.mark.integration
    def test_raises_type_error_if_predict_proba_does_not_exist(self, tmp_path):
        # Create a pipeline without .predict_proba() method (Regressor instead of Classifier)
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("rf_regressor", RandomForestRegressor())
        ])
        # Save pipeline to temporary file
        pipeline_path = tmp_path / "pipeline.joblib"
        joblib.dump(pipeline, pipeline_path)

        # Ensure .load_pipeline() raises TypeError
        with pytest.raises(TypeError) as exc_info:
            load_pipeline(pipeline_path)
        # Ensure error message is as expected
        error_msg = str(exc_info.value)
        assert "Error when loading pipeline"
        assert "Loaded pipeline does not have a .predict_proba() method" in error_msg


# --- FastAPI endpoint "/predict" ---
client = TestClient(app)

class TestPredict:
    # Valid input returns HTTP 200 OK status, a valid response schema and probabilities sum to 1
    @pytest.mark.integration
    @pytest.mark.parametrize("valid_input", [
        # Single input dict
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
        # Batch input List[dict]
        [
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
    ])
    def test_happy_path(self, valid_input):
        # Post request to predict endpoint
        response = client.post("/predict", json=valid_input)
        prediction_response = response.json()

        # Ensure post request was successful
        assert response.status_code == 200
        # Ensure prediction response has expected schema 
        assert "results" in prediction_response
        assert isinstance(prediction_response["results"], list)
        assert "n_predictions" in prediction_response
        assert isinstance(prediction_response["n_predictions"], int)
        # Ensure prediction response has expected number of results
        if isinstance(valid_input, list):  # batch input list
            expected_n_results = len(valid_input)
        else:  # single input dict
            expected_n_results = 1
        assert len(prediction_response["results"]) == expected_n_results
        assert prediction_response["n_predictions"] == expected_n_results
        # Ensure each prediction result has expected schema
        for prediction_result in prediction_response["results"]:
            assert isinstance(prediction_result, dict)
            assert "prediction" in prediction_result
            assert "probabilities" in prediction_result
            # Ensure prediction has expected schema
            prediction = prediction_result["prediction"]
            assert isinstance(prediction, str)
            assert prediction in ["Default", "No Default"]
            # Ensure probabilities has expected schema
            predicted_probabilities = prediction_result["probabilities"]
            assert isinstance(predicted_probabilities, dict)
            assert "Default" in predicted_probabilities
            assert "No Default" in predicted_probabilities
            assert isinstance(predicted_probabilities["Default"], float)
            assert isinstance(predicted_probabilities["No Default"], float)
            # Ensure probabilities sum to approximately 1
            assert (predicted_probabilities["Default"] + predicted_probabilities["No Default"]) == pytest.approx(1.0)  # default relative tolerance of 1e-6 (0.0001%) and absolute tolerance of 1e-12

    # Empty batch input
    @pytest.mark.integration
    def test_empty_batch_input_happy_path(self):
        empty_batch_input = []
        
        # Post request to predict endpoint
        response = client.post("/predict", json=empty_batch_input)

        # Ensure post request is successful
        assert response.status_code == 200
        # Ensure prediction response is as expected
        prediction_response = response.json()
        expected_prediction_response = {
            "results": [],
            "n_predictions": 0
        }
        assert prediction_response == expected_prediction_response

    # Pydantic validation error
    @pytest.mark.integration
    @pytest.mark.parametrize("invalid_input", [
        # Empty single input
        {}, 
        # Batch input with empty dictionaries
        [{}, {}], 
        # Missing required field  
        {
            "income": 300000,
            # "age" field missing 
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
        # Invalid data type
        {
            "income": 300000,
            "age": "a string",  # invalid type
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
        # Missing value in a required field
        {
            "income": 300000,
            "age": None,  # missing value 
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
        # Invalid string Enum 
        {
            "income": 300000,
            "age": 30,   
            "experience": 3,
            "married": "single",
            "house_ownership": "hopefully_in_the_future",  # invalid Enum
            "car_ownership": "no",
            "profession": "artist",
            "city": "sikar",
            "state": "rajasthan",
            "current_job_yrs": 3,
            "current_house_yrs": 11           
        },
        # Out-of-range value
        {
            "income": 300000,
            "age": 999,  # out-of-range value 
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
    ])
    def test_return_http_422_for_pydantic_validation_error(self, invalid_input):
        # Make post request to predict endpoint 
        response = client.post("/predict", json=invalid_input)  

        # Ensure response has status code 422 (Unprocessable Entity)
        assert response.status_code == 422
        # Ensure error location of all errors contains the response body
        error_detail = response.json()["detail"]
        assert all("body" in error["loc"] for error in error_detail)
        # Ensure error location of all errors contains either PipelineInput or list[PipelineInput] pydantic model 
        assert all(
            any(input in error["loc"] for input in ["PipelineInput", "list[PipelineInput]"]) 
            for error in error_detail
        )
    
    # Extra field in input
    @pytest.mark.integration
    def test_input_with_and_without_extra_field_predict_same_result(self):
        valid_input = {
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
        valid_input_with_extra_field = {
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
            "current_house_yrs": 11,
            "extra_field": "should be ignored"  # extra field, otherwise identical      
        }

        # Post request to predict endpoint
        response = client.post("/predict", json=[valid_input, valid_input_with_extra_field])
        prediction_response = response.json()

        # Ensure post request was successful
        assert response.status_code == 200
        # Ensure prediction result for input with and without extra field are identical
        assert prediction_response["results"][0] == prediction_response["results"][1]

    # Optional field or its value is missing
    @pytest.mark.integration
    @pytest.mark.parametrize("missing_optional_input", [
        # Missing field
        {
            "income": 300000,
            "age": 30,
            "experience": 3,
            # "married" field missing, should impute the mode "single"
            "house_ownership": "rented",
            "car_ownership": "no",
            "profession": "artist",
            "city": "sikar",
            "state": "rajasthan",
            "current_job_yrs": 3,
            "current_house_yrs": 11
        },
        # Field is present, but its value is missing
        {
            "income": 300000,
            "age": 30,
            "experience": 3,
            "married": None,  # missing value, should impute the mode "single"
            "house_ownership": "rented",
            "car_ownership": "no",
            "profession": "artist",
            "city": "sikar",
            "state": "rajasthan",
            "current_job_yrs": 3,
            "current_house_yrs": 11
        }
    ])
    def test_input_with_and_without_missing_optional_input_predict_same_result(self, missing_optional_input):
        input = {
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

        # Post request to predict endpoint
        response = client.post("/predict", json=[input, missing_optional_input])
        prediction_response = response.json()

        # Ensure post request was successful
        assert response.status_code == 200
        # Ensure prediction result for input with and without missing optional field is identical
        assert prediction_response["results"][0] == prediction_response["results"][1]

    # Rounding float to int
    @pytest.mark.integration
    def test_rounding_float_vs_int_input_predict_same_result(self):
        int_input = {
            "income": 300000,  # int
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
        float_input = {
            "income": 300000.4,  # float, otherwise identical
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

        # Post request to predict endpoint
        response = client.post("/predict", json=[int_input, float_input])
        prediction_response = response.json()

        # Ensure post request was successful
        assert response.status_code == 200
        # Ensure prediction result for int and float input is identical
        assert prediction_response["results"][0] == prediction_response["results"][1]

    # Internal server error
    @pytest.mark.integration
    def test_return_http_500_for_pipeline_failure(self, monkeypatch, request):
        # --- Create a faulty pipeline ---
        # Load the real pipeline
        root_dir = Path(str(request.config.rootdir))
        pipeline_path = root_dir / "models" / "loan_default_rf_pipeline.joblib"
        real_pipeline = joblib.load(pipeline_path)

        # Get the original job stability map and create a faulty version by removing a category
        original_map = real_pipeline.named_steps["job_stability_transformer"].job_stability_map
        faulty_map = original_map.copy()
        del faulty_map["firefighter"]

        # Create a new pipeline instance with the faulty map
        faulty_pipeline = real_pipeline.set_params(job_stability_transformer__job_stability_map=faulty_map)

        # --- Inject the faulty pipeline into the application ---
        monkeypatch.setattr("backend.app.pipeline", faulty_pipeline)

        # --- Post request with unknown "firefighter" category ---
        input_with_unknown_category = {
            "income": 300000,
            "age": 30,
            "experience": 3,
            "married": "single",
            "house_ownership": "rented",
            "car_ownership": "no",
            "profession": "firefighter",  # unknown category will cause pipeline failure
            "city": "sikar",
            "state": "rajasthan",
            "current_job_yrs": 3,
            "current_house_yrs": 11
        }

        response = client.post("/predict", json=input_with_unknown_category)

        # Ensure response has status code 500 (Internal Server Error)
        assert response.status_code == 500
        # Ensure error detail is as expected
        assert "Internal server error during loan default prediction" in response.text

    # --- Model Behavior Tests ---
    # Low risk features predict low default probability    
    @pytest.mark.integration
    def test_low_risk_features_predict_low_default_probability(self):
        low_risk_input = { 
                "income": 8_000_000,
                "age": 30,
                "experience": 10,
                "married": "married",
                "house_ownership": "rented",
                "car_ownership": "yes",
                "profession": "architect",
                "city": "delhi_city",
                "state": "assam",
                "current_job_yrs": 10,
                "current_house_yrs": 14           
            }

        # Post request to predict endpoint
        response = client.post("/predict", json=low_risk_input)
        prediction_response = response.json()

        # Ensure post request was successful
        assert response.status_code == 200
        # Ensure probability of default is low (less than 10%)
        prob_default = prediction_response["results"][0]["probabilities"]["Default"]
        assert prob_default < 0.1

    # High compared to low income predicts lower default probability
    @pytest.mark.integration
    def test_high_vs_low_income_predicts_lower_default_probability(self):
        batch_input = [
            # high income
            { 
                "income": 8_000_000,
                "age": 50,
                "experience": 3,
                "married": "single",
                "house_ownership": "owned",
                "car_ownership": "no",
                "profession": "artist",
                "city": "sikar",
                "state": "rajasthan",
                "current_job_yrs": 3,
                "current_house_yrs": 11            
            },
            # low income, otherwise identical
            { 
                "income": 800_000,
                "age": 50,
                "experience": 3,
                "married": "single",
                "house_ownership": "owned",
                "car_ownership": "no",
                "profession": "artist",
                "city": "sikar",
                "state": "rajasthan",
                "current_job_yrs": 3,
                "current_house_yrs": 11           
            }
        ]

        # Post request to predict endpoint
        response = client.post("/predict", json=batch_input)
        prediction_response = response.json()

        # Ensure post request was successful
        assert response.status_code == 200
        # Ensure probability of default is lower for house owned compared to rented
        high_income_default_prob = prediction_response["results"][0]["probabilities"]["Default"]
        low_income_default_prob = prediction_response["results"][1]["probabilities"]["Default"]
        assert high_income_default_prob < low_income_default_prob 

    # Low compared to high risk age predicts lower default probability
    @pytest.mark.integration
    def test_low_vs_high_risk_age_predicts_lower_default_probability(self):
        batch_input = [
            # low risk age
            { 
                "income": 8_000_000,
                "age": 30,  
                "experience": 10,
                "married": "married",
                "house_ownership": "rented",
                "car_ownership": "yes",
                "profession": "architect",
                "city": "delhi_city",
                "state": "assam",
                "current_job_yrs": 10,
                "current_house_yrs": 14           
            },
            # high risk age, otherwise identical
            { 
                "income": 8_000_000,
                "age": 75,
                "experience": 10,
                "married": "married",
                "house_ownership": "rented",
                "car_ownership": "yes",
                "profession": "architect",
                "city": "delhi_city",
                "state": "assam",
                "current_job_yrs": 10,
                "current_house_yrs": 14           
            }
        ]

        # Post request to predict endpoint
        response = client.post("/predict", json=batch_input)
        prediction_response = response.json()

        # Ensure post request was successful
        assert response.status_code == 200
        # Ensure probability of default is higher for high compared to low risk age
        low_risk_age_default_prob = prediction_response["results"][0]["probabilities"]["Default"]
        high_risk_age_default_prob = prediction_response["results"][1]["probabilities"]["Default"]
        assert low_risk_age_default_prob < high_risk_age_default_prob 

    # House owned compared to rented predicts lower default probability
    @pytest.mark.integration
    def test_house_owned_vs_rented_predicts_lower_default_probability(self):
        batch_input = [
            # house owned
            { 
                "income": 1_000_000,
                "age": 50,
                "experience": 3,
                "married": "single",
                "house_ownership": "owned",
                "car_ownership": "no",
                "profession": "artist",
                "city": "sikar",
                "state": "rajasthan",
                "current_job_yrs": 3,
                "current_house_yrs": 11            
            },
            # house rented, otherwise identical
            { 
                "income": 1_000_000,
                "age": 50,
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
        ]

        # Post request to predict endpoint
        response = client.post("/predict", json=batch_input)
        prediction_response = response.json()

        # Ensure post request was successful
        assert response.status_code == 200
        # Ensure probability of default is lower for house owned compared to rented
        house_owned_default_prob = prediction_response["results"][0]["probabilities"]["Default"]
        house_rented_default_prob = prediction_response["results"][1]["probabilities"]["Default"]
        assert house_owned_default_prob < house_rented_default_prob 

    # High compared to low experience predicts lower default probability
    @pytest.mark.integration
    def test_high_vs_low_experience_predicts_lower_default_probability(self):
        batch_input = [
            # high experience
            { 
                "income": 1_000_000,
                "age": 50,
                "experience": 15,
                "married": "single",
                "house_ownership": "owned",
                "car_ownership": "no",
                "profession": "artist",
                "city": "sikar",
                "state": "rajasthan",
                "current_job_yrs": 3,
                "current_house_yrs": 11            
            },
            # low experience, otherwise identical
            { 
                "income": 1_000_000,
                "age": 50,
                "experience": 0,
                "married": "single",
                "house_ownership": "owned",
                "car_ownership": "no",
                "profession": "artist",
                "city": "sikar",
                "state": "rajasthan",
                "current_job_yrs": 3,
                "current_house_yrs": 11           
            }
        ]

        # Post request to predict endpoint
        response = client.post("/predict", json=batch_input)
        prediction_response = response.json()

        # Ensure post request was successful
        assert response.status_code == 200
        # Ensure probability of default is lower for house owned compared to rented
        high_experience_default_prob = prediction_response["results"][0]["probabilities"]["Default"]
        low_experience_default_prob = prediction_response["results"][1]["probabilities"]["Default"]
        assert high_experience_default_prob < low_experience_default_prob 
