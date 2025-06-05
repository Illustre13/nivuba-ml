# --- Imports ---
# Standard library imports
import warnings
from unittest.mock import patch

# Third-party library imports
import pytest
from fastapi.testclient import TestClient

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Local imports
from frontend.app import predict_loan_default
from backend.app import app

# --- Test Setup ---
# Create FastAPI test client to be used in tests
client = TestClient(app)        

# Function to redirect post requests to the test client (use with mock_post_request.side_effect)
def redirect_post_request_to_testclient(url, json, timeout):  # parameters must mirror mocked post request in .predict_loan_default()
    # Use the JSON that .predict_loan_default() creates, ignore "url" and "timeout"
    return client.post("/predict", json=json)


# --- Function .predict_loan_default() ---
class TestPredictLoanDefault:
    # Happy path
    @pytest.mark.integration
    @patch("frontend.app.requests.post")
    def test_happy_path(self, mock_post_request):
        # Raw inputs from Gradio UI as dictionary
        inputs = {
            "age": 30, 
            "married": "Single", 
            "income": 300000, 
            "car_ownership": "No", 
            "house_ownership": "Neither Rented Nor Owned", 
            "current_house_yrs": 11, 
            "city": "Sikar", 
            "state": "Rajasthan", 
            "profession": "Artist", 
            "experience": 3, 
            "current_job_yrs": 3
        }
        # Mock the post request to redirect it to the FastAPI test client
        mock_post_request.side_effect = redirect_post_request_to_testclient

        # Call .predict_loan_default()
        prediction, probabilities = predict_loan_default(**inputs)

        # Ensure requests.post() was called once
        mock_post_request.assert_called_once()
        # Ensure prediction is as expected
        assert isinstance(prediction, str)
        assert prediction in ["Default", "No Default"]
        # Ensure probabilities is as expected
        assert isinstance(probabilities, dict)
        assert "Default" in probabilities
        assert "No Default" in probabilities
        assert isinstance(probabilities["Default"], float)
        assert isinstance(probabilities["No Default"], float)
        # Ensure probabilities sum to approximately 1
        assert (probabilities["Default"] + probabilities["No Default"]) == pytest.approx(1.0)  # default relative tolerance of 1e-6 (0.0001%) and absolute tolerance of 1e-12

    # HTTP 422 Pydantic validation error
    @pytest.mark.integration
    @pytest.mark.parametrize("field, invalid_value, partial_error_msg", [
        # Test case 1: Out-of-range value in numeric field
        ("age", 5, "Age: Enter a number between 21 and 79."),
        # Test case 2: Invalid Enum in string field
        ("married", "invalid value", "Married/Single: Select 'Married' or 'Single'"),
        # Test case 3: Wrong type (e.g. string instead of number)
        ("income", "300000", "Income: Enter a number that is 0 or greater.")
    ])
    @patch("frontend.app.requests.post")
    def test_http_422_pydantic_validation_error(self, mock_post_request, field, invalid_value, partial_error_msg, caplog):
        # Invalid inputs from Gradio UI
        invalid_inputs = {
            "age": 30, 
            "married": "Single",  
            "income": 300000, 
            "car_ownership": "No", 
            "house_ownership": "Neither Rented Nor Owned", 
            "current_house_yrs": 11, 
            "city": "Sikar", 
            "state": "Rajasthan", 
            "profession": "Artist", 
            "experience": 3, 
            "current_job_yrs": 3
        }
        invalid_inputs[field] = invalid_value
        # Mock the post request to redirect it to the FastAPI test client
        mock_post_request.side_effect = redirect_post_request_to_testclient

        # Call .predict_loan_default()
        prediction, probabilities = predict_loan_default(**invalid_inputs)

        # Ensure expected error messages for Gradio frontend
        assert prediction == f"Input Error! Please check your inputs and try again.\n{partial_error_msg}\n"
        assert probabilities == ""
        # Ensure exactly one error was logged with correct level and message
        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "WARNING"
        assert "Received 422 validation error from backend" in caplog.records[0].message
