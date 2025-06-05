# --- Imports ---
# Standard library imports
from typing import List, Dict, Any, Callable
from enum import Enum

# Third-party library imports
import pytest
from pydantic import ValidationError 

# Local imports
from backend.schemas import (
    MarriedEnum,
    HouseOwnershipEnum,
    CarOwnershipEnum,
    ProfessionEnum,
    CityEnum,
    StateEnum,
    PredictionEnum,
    PipelineInput,
    PredictedProbabilities,
    PredictionResult,
    PredictionResponse
)
from src.global_constants import (
    MARRIED_LABELS,
    HOUSE_OWNERSHIP_LABELS,
    CAR_OWNERSHIP_LABELS,
    PROFESSION_LABELS,
    CITY_LABELS,
    STATE_LABELS
)

# --- Constants ----
REQUIRED_FIELDS: List[str] = [
    "income", "age", "experience", "profession", "city", 
    "state", "current_job_yrs", "current_house_yrs"
]
OPTIONAL_FIELDS: List[str] = ["married", "house_ownership", "car_ownership"]


# --- Fixtures ----
# Define valid pipeline input for testing 
@pytest.fixture
def valid_pipeline_input() -> Dict[str, Any]:
    return {
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


# --- Enums ---
# Basic tests for all string Enum classes
class TestEnums:
    # Correct string labels
    @pytest.mark.unit
    @pytest.mark.parametrize("enum_class, expected_string_labels", [
        (MarriedEnum, MARRIED_LABELS),
        (HouseOwnershipEnum, HOUSE_OWNERSHIP_LABELS),
        (CarOwnershipEnum, CAR_OWNERSHIP_LABELS),
        (ProfessionEnum, PROFESSION_LABELS),
        (CityEnum, CITY_LABELS),
        (StateEnum, STATE_LABELS),
        (PredictionEnum, ["Default", "No Default"])
    ])
    def test_enum_contains_all_string_labels(
        self, 
        enum_class: Enum, 
        expected_string_labels: List[str]
    ) -> None:
        enum_values = {member.value for member in enum_class}
        missing_values = set(expected_string_labels) - enum_values
        extra_values = enum_values - set(expected_string_labels)
        assert not missing_values, f"Missing values in {enum_class.__name__}: {missing_values}"
        assert not extra_values, f"Extra values in {enum_class.__name__}: {extra_values}"

    # Invalid values (wrong string, empty string, None, wrong data type)
    @pytest.mark.unit
    @pytest.mark.parametrize("enum_class", [
        MarriedEnum,
        HouseOwnershipEnum,
        CarOwnershipEnum,
        ProfessionEnum,
        CityEnum,
        StateEnum,
        PredictionEnum
    ])
    @pytest.mark.parametrize("invalid_value", [
        "wrong string that is not an Enum member",  
        "",  
        None,  
        1,
        1.23,
        False,
        ["a", "list"],
        ("a", "tuple"),
        {"a": "dictionary"},
        {"a", "set"}
    ])
    def test_raises_value_error_for_invalid_value(self, enum_class: Enum, invalid_value: Any) -> None:
        with pytest.raises(ValueError):
            enum_class(invalid_value)

    # Wrong casing (Enum values are snake_case)
    @pytest.mark.unit
    @pytest.mark.parametrize("enum_class, enum_values", [
        (MarriedEnum, MARRIED_LABELS),
        (HouseOwnershipEnum, HOUSE_OWNERSHIP_LABELS),
        (CarOwnershipEnum, CAR_OWNERSHIP_LABELS),
        (ProfessionEnum, PROFESSION_LABELS),
        (CityEnum, CITY_LABELS),
        (StateEnum, STATE_LABELS)
    ])
    @pytest.mark.parametrize("wrong_casing", [
        str.upper,
        str.title,
        str.capitalize
    ])
    def test_raises_value_error_for_wrong_casing(
        self, 
        enum_class: Enum, 
        enum_values: List[str], 
        wrong_casing: Callable[[str], str]
    ) -> None:
        for enum_value in enum_values:
            value_with_wrong_casing = wrong_casing(enum_value)
            with pytest.raises(ValueError):
                enum_class(value_with_wrong_casing)

# PredictionEnum
class TestPredictionEnum:
    # Happy path
    @pytest.mark.unit
    @pytest.mark.parametrize("input_string, expected_enum", [
        ("Default", PredictionEnum.DEFAULT),
        ("No Default", PredictionEnum.NO_DEFAULT)
    ])
    def test_happy_path(self, input_string: str, expected_enum: PredictionEnum) -> None:
        prediction_enum = PredictionEnum(input_string)
        assert prediction_enum == expected_enum    

    # Invalid string values
    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_string", [
        "Yes",
        "No",
        "default",     # wrong case (lowercase)
        "DEFAULT",     # wrong case (uppercase)
        "no default",  # wrong case (lowercase)
        "NO DEFAULT",  # wrong case (uppercase)
        "no_default",  # wrong separator and case (snake case)
        "No_default",  # wrong separator and case (captical case)
    ])
    def test_raises_value_error_for_invalid_string(self, invalid_string: str) -> None:
        with pytest.raises(ValueError):
            PredictionEnum(invalid_string)


# --- Pydantic Model: PipelineInput ---
class TestPipelineInput:
    # Happy path
    @pytest.mark.unit
    def test_happy_path(self, valid_pipeline_input: Dict[str, Any]) -> None:
        pipeline_input = PipelineInput(**valid_pipeline_input)
        assert pipeline_input.income == 1000000
        assert pipeline_input.age == 30
        assert pipeline_input.experience == 10
        assert pipeline_input.married == "married"
        assert pipeline_input.house_ownership == "rented"
        assert pipeline_input.car_ownership == "yes"
        assert pipeline_input.profession == "architect"
        assert pipeline_input.city == "delhi_city"
        assert pipeline_input.state == "assam"
        assert pipeline_input.current_job_yrs == 7
        assert pipeline_input.current_house_yrs == 12

    # Field validator: Convert float to int
    @pytest.mark.unit
    @pytest.mark.parametrize("field, input_value, expected_value", [
        ("income", 100000.6, 100001),  # round up
        ("income", 100000.4, 100000),  # round down
        ("age", 25.5, 26),  # round up for odd numbers
        ("age", 26.5, 26),  # round down for even numbers (banker's rounding)
        ("experience", 10.0, 10),  # whole float
        ("experience", 0.0, 0),  
        ("current_job_yrs", 5, 5),  # passthrough int
        ("current_job_yrs", 14, 14),  
        ("current_house_yrs", 10.2, 10),  # round at lower boundary
        ("current_house_yrs", 13.9, 14),  # round at upper boundary
    ])
    def test_converts_float_to_int(
        self,
        valid_pipeline_input: Dict[str, Any], 
        field: str, 
        input_value: float | int, 
        expected_value: int
    ) -> None:
        pipeline_input_dict = valid_pipeline_input.copy()
        pipeline_input_dict[field] = input_value
        pipeline_input = PipelineInput(**pipeline_input_dict)
        assert getattr(pipeline_input, field) == expected_value

    # Extra field 
    @pytest.mark.unit
    def test_extra_field_is_ignored(self, valid_pipeline_input: Dict[str, Any]) -> None:
        input_with_extra_field = valid_pipeline_input.copy()
        input_with_extra_field["extra_field"] = "should be ignored"
        pipeline_input = PipelineInput(**input_with_extra_field)
        # Ensure extra field is not present
        assert not hasattr(pipeline_input, "extra_field")
        # Ensure instance is as expected
        assert pipeline_input == PipelineInput(**valid_pipeline_input)

    # Missing required field
    @pytest.mark.unit 
    @pytest.mark.parametrize("required_field", REQUIRED_FIELDS)
    def test_raises_validation_error_if_required_field_is_missing(
            self, 
            valid_pipeline_input: Dict[str, Any],
            required_field: str
    ) -> None:
        pipeline_input_with_missing_required_field = valid_pipeline_input.copy()
        del pipeline_input_with_missing_required_field[required_field]
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PipelineInput(**pipeline_input_with_missing_required_field)
        errors = exc_info.value.errors()
        # Ensure error location of all errors is the required field we are testing
        assert all(error["loc"][0] == required_field for error in errors)
        # Ensure error type of at least one error is "missing"
        assert any(error["type"] == "missing" for error in errors)

    # Missing optional field
    @pytest.mark.unit 
    @pytest.mark.parametrize("optional_field", OPTIONAL_FIELDS)
    def test_assigns_none_if_optional_field_is_missing(
            self, 
            valid_pipeline_input: Dict[str, Any],
            optional_field: str
    ) -> None:
        pipeline_input_with_missing_optional_field = valid_pipeline_input.copy()
        del pipeline_input_with_missing_optional_field[optional_field]
        pipeline_input = PipelineInput(**pipeline_input_with_missing_optional_field)
        assert hasattr(pipeline_input, optional_field)
        assert pipeline_input.model_dump()[optional_field] is None

    # Missing value in required field
    @pytest.mark.unit 
    @pytest.mark.parametrize("required_field", REQUIRED_FIELDS)
    def test_raises_validation_error_if_required_field_is_none(
            self, 
            valid_pipeline_input: Dict[str, Any],
            required_field: str
    ) -> None:
        pipeline_input_with_missing_required_value = valid_pipeline_input.copy()
        pipeline_input_with_missing_required_value[required_field] = None
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PipelineInput(**pipeline_input_with_missing_required_value)
        errors = exc_info.value.errors()
        # Ensure error location of all errors is the required field we are testing
        assert all(error["loc"][0] == required_field for error in errors)
        # Ensure error type of at least one error is "int_type", "float_type" or "enum" (which take precedence over "none_forbidden")
        assert any(error["type"] in ["int_type", "float_type", "enum"] for error in errors)

    # Missing value in optional field
    @pytest.mark.unit 
    @pytest.mark.parametrize("optional_field", OPTIONAL_FIELDS)
    def test_assigns_none_if_optional_field_is_none(
            self, 
            valid_pipeline_input: Dict[str, Any],
            optional_field: str
    ) -> None:
        pipeline_input_with_missing_optional_value = valid_pipeline_input.copy()
        pipeline_input_with_missing_optional_value[optional_field] = None
        pipeline_input = PipelineInput(**pipeline_input_with_missing_optional_value)
        assert pipeline_input.model_dump()[optional_field] is None

    # Wrong data type of string fields
    @pytest.mark.unit 
    @pytest.mark.parametrize("string_field", ["married", "house_ownership", "car_ownership", "profession", "city", "state"])
    @pytest.mark.parametrize("wrong_data_type", [
        1,
        1.23,
        False,
        ["a", "list"],
        ("a", "tuple"),
        {"a": "dictionary"},
        {"a", "set"}
    ])
    def test_raises_validation_error_if_string_field_has_wrong_type(
            self, 
            valid_pipeline_input: Dict[str, Any],
            string_field: str, 
            wrong_data_type: Any
    ) -> None:
        pipeline_input_with_wrong_type = valid_pipeline_input.copy()
        pipeline_input_with_wrong_type[string_field] = wrong_data_type
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PipelineInput(**pipeline_input_with_wrong_type)
        errors = exc_info.value.errors()
        # Ensure error location of all errors is the string field we are testing
        assert all(error["loc"][0] == string_field for error in errors)
        # Ensure error type of at least one error is "enum" 
        assert any(error["type"] == "enum" for error in errors)

    # Wrong data type of numeric fields
    @pytest.mark.unit 
    @pytest.mark.parametrize("numeric_field", ["income", "age", "experience", "current_job_yrs", "current_house_yrs"])
    @pytest.mark.parametrize("wrong_data_type", [
        "a string",
        "1.23",
        False,
        ["a", "list"],
        ("a", "tuple"),
        {"a": "dictionary"},
        {"a", "set"}
    ])
    def test_raises_validation_error_if_numeric_field_has_wrong_type(
            self, 
            valid_pipeline_input: Dict[str, Any],
            numeric_field: str, 
            wrong_data_type: Any
    ) -> None:
        pipeline_input_with_wrong_type = valid_pipeline_input.copy()
        pipeline_input_with_wrong_type[numeric_field] = wrong_data_type
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PipelineInput(**pipeline_input_with_wrong_type)
        errors = exc_info.value.errors()
        # Ensure error location of all errors is the numeric field we are testing
        assert all(error["loc"][0] == numeric_field for error in errors)
        # Ensure error type of at least one error is "int_type" or "float_type" (due to strict mode)
        assert any(error["type"] in ["int_type", "float_type"] for error in errors)

    # Out-of-range numeric values
    @pytest.mark.unit 
    @pytest.mark.parametrize("numeric_field, oor_value", [
        ("income", -50),  # negative 
        ("income", -0.01),  # below minimum 
        ("age", -50),  # negative  
        ("age", 0),  # zero
        ("age", 20.99),  # below minimum 
        ("age", 79.01),  # above maximum
        ("age", 1000),  # large number
        ("experience", -50),  # negative
        ("experience", -0.01),  # below minimum
        ("experience", 20.01),  # above minimum
        ("experience", 1000),  # large number
        ("current_job_yrs", -50),  # negative
        ("current_job_yrs", -0.01),  # below minimum
        ("current_job_yrs", 14.01),  # above maximum
        ("current_job_yrs", 1000),  # large number
        ("current_house_yrs", -50),  # negative
        ("current_house_yrs", 0),  # zero
        ("current_house_yrs", 9.99),  # below minimum
        ("current_house_yrs", 14.01),  # above maximum
        ("current_house_yrs", 1000),  # large number
    ])
    def test_raises_validation_error_if_numeric_value_is_out_of_range(
            self, 
            valid_pipeline_input: Dict[str, Any],
            numeric_field: str, 
            oor_value: int | float
    ) -> None:
        pipeline_input_with_oor_value = valid_pipeline_input.copy()
        pipeline_input_with_oor_value[numeric_field] = oor_value
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PipelineInput(**pipeline_input_with_oor_value)
        errors = exc_info.value.errors()
        # Ensure error location of all errors is the numeric field we are testing
        assert all(error["loc"][0] == numeric_field for error in errors)
        # Ensure error type of at least one error is "greater_than_equal" or "less_than_equal"
        assert any(error["type"] in ["greater_than_equal", "less_than_equal"] for error in errors)

    # Boundary numeric values
    @pytest.mark.unit
    @pytest.mark.parametrize("numeric_field, boundary_value", [
        ("income", 0), ("income", 0.0),  # minimum 
        ("age", 21),  ("age", 21.0),  # minimum 
        ("age", 79),  ("age", 79.0),  # maximum
        ("experience", 0),  ("experience", 0.0), 
        ("experience", 20),  ("experience", 20.0), 
        ("current_job_yrs", 0),  ("current_job_yrs", 0.0), 
        ("current_job_yrs", 14),  ("current_job_yrs", 14.0), 
        ("current_house_yrs", 10),  ("current_house_yrs", 10.0), 
        ("current_house_yrs", 14),  ("current_house_yrs", 14.0), 
    ])
    def test_boundary_values_are_valid_for_numeric_fields(
        self, 
        valid_pipeline_input: Dict[str, Any],
        numeric_field: str,
        boundary_value: int | float
    ) -> None:
        pipeline_input_with_boundary_value = valid_pipeline_input.copy()
        pipeline_input_with_boundary_value[numeric_field] = boundary_value
        pipeline_input = PipelineInput(**pipeline_input_with_boundary_value)
        value = getattr(pipeline_input, numeric_field)
        expected_value = int(boundary_value)
        assert isinstance(value, int)
        assert value == expected_value

    # Invalid string enum values
    @pytest.mark.unit 
    @pytest.mark.parametrize("string_field, invalid_string_enum", [
        ("married", "divorced"), 
        ("married", "yes"), 
        ("married", "no"), 
        ("married", "Single"),  # wrong casing
        ("married", "SINGLE"),  # wrong casing
        ("married", "Married"),  # wrong casing
        ("married", "MARRIED"),  # wrong casing
        ("house_ownership", "maybe"), 
        ("house_ownership", "yes"), 
        ("house_ownership", "no"), 
        ("house_ownership", "mortgaged"), 
        ("house_ownership", "hopefully_in_the_future"), 
        ("house_ownership", "Rented"),  # wrong casing
        ("house_ownership", "OWNED"),  # wrong casing
        ("house_ownership", "Norent_Noown"),  # wrong casing
        ("car_ownership", "maybe"), 
        ("car_ownership", "lamborghini"), 
        ("car_ownership", "soon"), 
        ("car_ownership", "Yes"),  # wrong casing
        ("car_ownership", "NO"),  # wrong casing
        ("profession", "unknown"), 
        ("profession", "jedi_knight"), 
        ("profession", "princess"), 
        ("profession", "divorce_lawyer"), 
        ("profession", "Air_Traffic_Controller"),  # wrong casing
        ("profession", "Army_officer"),  # wrong casing
        ("city", "unknown"), 
        ("city", "metropolis"), 
        ("city", "new_york"), 
        ("city", "tokyo"), 
        ("city", "Chandigarh_City"),  # wrong casing
        ("city", "ADONI"),  # wrong casing
        ("state", "unknown"), 
        ("state", "india"), 
        ("state", "california"), 
        ("state", "Andhra_Pradesh"),  # wrong casing 
        ("state", "ASSAM"),  # wrong casing 
    ])
    def test_raises_validation_error_for_invalid_string_enum(
            self, 
            valid_pipeline_input: Dict[str, Any],
            string_field: str, 
            invalid_string_enum: str 
    ) -> None:
        pipeline_input_with_invalid_string = valid_pipeline_input.copy()
        pipeline_input_with_invalid_string[string_field] = invalid_string_enum
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PipelineInput(**pipeline_input_with_invalid_string)
        errors = exc_info.value.errors()
        # Ensure error location of all errors is the string field we are testing
        assert all(error["loc"][0] == string_field for error in errors)
        # Ensure error type of all errors is "enum"
        assert all(error["type"] == "enum" for error in errors)

    # Valid string enum values
    @pytest.mark.unit 
    @pytest.mark.parametrize("string_field, valid_string, expected_enum_member", [
        ("married", "single", MarriedEnum.SINGLE), 
        ("house_ownership", "rented", HouseOwnershipEnum.RENTED),    
        ("car_ownership", "yes", CarOwnershipEnum.YES), 
        ("profession", "air_traffic_controller", ProfessionEnum.AIR_TRAFFIC_CONTROLLER), 
        ("city", "adoni", CityEnum.ADONI), 
        ("state", "andhra_pradesh", StateEnum.ANDHRA_PRADESH)
    ])
    def test_accepts_valid_string_enum(
            self, 
            valid_pipeline_input: Dict[str, Any],
            string_field: str, 
            valid_string: str,
            expected_enum_member: Enum
    ) -> None:
        pipeline_input_with_valid_string = valid_pipeline_input.copy()
        pipeline_input_with_valid_string[string_field] = valid_string
        pipeline_input = PipelineInput(**pipeline_input_with_valid_string)
        enum_member = getattr(pipeline_input, string_field)
        assert enum_member == expected_enum_member


# --- Pydantic Model: PredictedProbabilities ---
class TestPredictedProbabilities:
    # Instantiation happy path
    @pytest.mark.unit
    def test_instantiation_happy_path(self) -> None:
        valid_input: Dict[str, float] = {
            "default": 0.8,
            "no_default": 0.2
        }
        predicted_probabilities = PredictedProbabilities(**valid_input)
        assert predicted_probabilities.default == 0.8
        assert predicted_probabilities.no_default == 0.2

    # Serialization alias
    @pytest.mark.unit
    def test_serialization_alias(self) -> None:
        predicted_probabilities = PredictedProbabilities(default=0.8, no_default=0.2)
        output = predicted_probabilities.model_dump(by_alias=True)
        # Ensure "default" is serialized to "Default"
        assert "Default" in output
        # Ensure "no_default" is serialized to "No Default"
        assert "No Default" in output 
        # Ensure entire output is as expected
        expected_output: Dict[str, float] = {
            "Default": 0.8,
            "No Default": 0.2
        }
        assert output == expected_output        

    # Field validator: Rounding is applied to all fields
    @pytest.mark.unit
    @pytest.mark.parametrize("field", ["default", "no_default"])
    def test_rounding_is_applied_to_all_fields(self, field: str) -> None:
        other_field = "default" if field == "no_default" else "no_default"
        valid_input = {
            field: 0.123456,
            other_field: 1 - 0.123456
        }
        predicted_probabilities = PredictedProbabilities(**valid_input)
        assert getattr(predicted_probabilities, field) == 0.123
    
    # Field validator: Rounding edge cases
    @pytest.mark.unit
    @pytest.mark.parametrize("input, expected_output", [
        (0.1234, 0.123),  # round down 
        (0.1236, 0.124),  # round up
        (0.12354, 0.124), 
        (0.12356, 0.124), 
        (0.0004, 0.0),  # round at bottom boundary
        (0.9996, 1.0),  # round at top boundary
        (0.1235, 0.123),  # stored as 0.12349999..., so rounds down
        (0.1245, 0.124),  # stored as 0.12449999..., so rounds down 
        (0.1255, 0.126),   # stored as 0.12550000000000003, so rounds up 
        (0.0005, 0.001),  # stored as 0.00050000000000000001..., so rounds up
        (0.9995, 1.0),  # stored as 0.99950000000000005..., so rounds up
    ])
    def test_rounding_edge_cases(
        self, 
        input: float,
        expected_output: float
    ) -> None:
        predicted_probabilities = PredictedProbabilities(default=input, no_default=1-input)
        assert predicted_probabilities.default == expected_output

    # Model validator: Probabilities sum to 1 happy path
    @pytest.mark.unit
    @pytest.mark.parametrize("prob_default, prob_no_default", [
        ([0.75, 0.25]), 
        ([0.5, 0.5]),         
        ([1.0, 0.0]),  # boundary
        ([0.0, 1.0]),  # boundary         
        ([0.999, 0.002]),  # 1.001 within tolerance          
        ([0.998, 0.001]),  # 0.999 within tolerance          
    ])
    def test_probabilities_sum_to_one_happy_path(
        self,
        prob_default: float,
        prob_no_default: float
    ) -> None:
        predicted_probabilities = PredictedProbabilities(default=prob_default, no_default=prob_no_default)
        assert predicted_probabilities.default == prob_default
        assert predicted_probabilities.no_default == prob_no_default

    # Model validator: Probabilities sum to 1 failure path
    @pytest.mark.unit
    @pytest.mark.parametrize("prob_default, prob_no_default", [
        ([0.8, 0.3]),  # 1.1 above     
        ([0.8, 0.1]),  # 0.9 below         
        ([1.0, 1.0]),  # 2.0 max above         
        ([0.0, 0.0]),  # 0.0 max below         
        ([0.999, 0.004]),  # 1.003 just above         
        ([0.996, 0.001]),  # 0.997 just below         
    ])
    def test_raises_value_error_if_probabilities_do_not_sum_to_one(
        self,
        prob_default: float,
        prob_no_default: float
    ) -> None:
        with pytest.raises(ValidationError) as exc_info:
            PredictedProbabilities(default=prob_default, no_default=prob_no_default)
        errors = exc_info.value.errors()
        # Ensure exactly one error
        assert len(errors) == 1
        # Ensure error location is an empty tuple (error in "whole model" not specific location)
        assert errors[0]["loc"] == ()
        # Ensure error type is "value_error"
        assert errors[0]["type"] == "value_error"
        # Ensure error message is as expected 
        assert "Probabilities must sum to 1.0" in errors[0]["msg"]  

    # Extra field 
    @pytest.mark.unit
    def test_extra_field_is_ignored(self) -> None:
        input_with_extra_field = {
            "default": 0.5, 
            "no_default": 0.5, 
            "extra_field": "should be ignored"
        }
        predicted_probabilities = PredictedProbabilities(**input_with_extra_field)
        # Ensure extra field is not present
        assert not hasattr(predicted_probabilities, "extra_field")
        # Ensure instance is as expected
        assert predicted_probabilities == PredictedProbabilities(default=0.5, no_default=0.5)

    # Missing field
    @pytest.mark.unit 
    @pytest.mark.parametrize("required_fields", [
        ["default"], 
        ["no_default"], 
        ["default", "no_default"]
    ])
    def test_raises_validation_error_if_fields_are_missing(
            self, 
            required_fields: List[str]
    ) -> None:
        input_with_missing_fields = {"default": 0.5, "no_default": 0.5}
        for required_field in required_fields:
            del input_with_missing_fields[required_field]
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PredictedProbabilities(**input_with_missing_fields)
        errors = exc_info.value.errors()
        # Ensure error location is the required field or fields
        assert {error["loc"][0] for error in errors} == set(required_fields)
        # Ensure error type of all errors is "missing"
        assert all(error["type"] == "missing" for error in errors)

    # None value
    @pytest.mark.unit 
    @pytest.mark.parametrize("fields", [
        ["default"], 
        ["no_default"],
        ["default", "no_default"]
    ])
    def test_raises_validation_error_if_fields_are_none(
            self, 
            fields: List[str]
    ) -> None:
        input_with_none = {"default": 0.5, "no_default": 0.5}
        for field in fields:
            input_with_none[field] = None
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PredictedProbabilities(**input_with_none)
        errors = exc_info.value.errors()
        # Ensure error location is the field or fields with a None value
        assert {error["loc"][0] for error in errors} == set(fields)
        # Ensure error type of all errors is "float_type" (which takes precedence over "none_forbidden")
        assert all(error["type"] == "float_type" for error in errors)

    # Wrong data type 
    @pytest.mark.unit 
    @pytest.mark.parametrize("float_fields", [
        ["default"], 
        ["no_default"],
        ["default", "no_default"]
    ])
    @pytest.mark.parametrize("wrong_data_type", [
        "a string",
        ["a", "list"],
        ("a", "tuple"),
        {"a": "dictionary"},
        {"a", "set"}
    ])
    def test_raises_validation_error_if_fields_have_wrong_type(
            self, 
            float_fields: List[str], 
            wrong_data_type: Any
    ) -> None:
        input_with_wrong_type = {"default": 0.5, "no_default": 0.5}
        for float_field in float_fields:
            input_with_wrong_type[float_field] = wrong_data_type
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PredictedProbabilities(**input_with_wrong_type)
        errors = exc_info.value.errors()
        # Ensure error location is the float field or fields we are testing
        assert {error["loc"][0] for error in errors} == set(float_fields) 
        # Ensure error type of all errors is "float_type" or "float_parsing" 
        assert all(error["type"] in ["float_type", "float_parsing"] for error in errors) 
  
    # Data type coersion 
    @pytest.mark.unit 
    @pytest.mark.parametrize("field", ["default", "no_default"])
    @pytest.mark.parametrize("coercible_input, expected_output", [
        ("0.123", 0.123),
        (0, 0.0),
        (True, 1.0),
        (False, 0.0),
    ])
    def test_coerces_valid_types_to_float(
            self, 
            field: str, 
            coercible_input: Any,
            expected_output: float
    ) -> None:
        other_field = "default" if field == "no_default" else "no_default"
        input_with_coercible_type = {
            field: coercible_input, 
            other_field: 1 - expected_output
        }
        predicted_probabilities = PredictedProbabilities(**input_with_coercible_type)
        assert getattr(predicted_probabilities, field) == expected_output

    # Out-of-range values
    @pytest.mark.unit 
    @pytest.mark.parametrize("float_field", ["default", "no_default"])
    @pytest.mark.parametrize("oor_value", [
        -0.001,  # just below minimum
        1.001,  # just above maximum
        float("nan"),  # not a number
        float("inf"),  # infinity
        float("-inf"),  # negative infinity
    ])
    def test_raises_validation_error_if_value_is_out_of_range(
            self, 
            float_field: str, 
            oor_value: float
    ) -> None:
        input_with_oor_value = {"default": 0.5, "no_default": 0.5}
        input_with_oor_value[float_field] = oor_value
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PredictedProbabilities(**input_with_oor_value)
        errors = exc_info.value.errors()
        # Ensure exactly one error
        assert len(errors) == 1
        # Ensure error location is the float field we are testing
        assert errors[0]["loc"][0] == float_field 
        # Ensure error type is "greater_than_equal" or "less_than_equal"
        assert errors[0]["type"] in ["greater_than_equal", "less_than_equal"]

    # Boundary values
    @pytest.mark.unit
    @pytest.mark.parametrize("field", ["default", "no_default"])
    @pytest.mark.parametrize("boundary_value", [0.0, 1.0])
    def test_boundary_values_are_valid(
        self, 
        field: str,
        boundary_value: float
    ) -> None:
        other_field = "default" if field == "no_default" else "no_default"
        input_with_boundary_value = {
            field: boundary_value,
            other_field: 1 - boundary_value
        }
        predicted_probabilities = PredictedProbabilities(**input_with_boundary_value)
        assert getattr(predicted_probabilities, field) == boundary_value


# --- Pydantic Model: PredictionResult ---
class TestPredictionResult:
    # Happy path
    @pytest.mark.unit
    def test_happy_path(self) -> None:
        valid_input: Dict[str, str | Dict[str, float]] = {
            "prediction": "Default",
            "probabilities": {
                "default": 0.8,
                "no_default": 0.2
            }
        }
        prediction_result = PredictionResult(**valid_input)
        assert prediction_result.prediction == PredictionEnum.DEFAULT
        assert prediction_result.probabilities == PredictedProbabilities(default=0.8, no_default=0.2)

    # Serialization aliases 
    @pytest.mark.unit
    def test_serialization_alias(self) -> None:
        prediction_result = PredictionResult(
            prediction=PredictionEnum.DEFAULT,
            probabilities=PredictedProbabilities(
                default=0.8,
                no_default=0.2
            )
        )
        output = prediction_result.model_dump(by_alias=True)
        expected_output: Dict[str, str | Dict[str, float]] = {
            "prediction": "Default",
            "probabilities": {
                "Default": 0.8,
                "No Default": 0.2
            }
        }
        assert output == expected_output 

    # Extra field 
    @pytest.mark.unit
    def test_extra_field_is_ignored(self) -> None:
        input_with_extra_field = {
            "prediction": "Default",
            "probabilities": {
                "default": 0.5,
                "no_default": 0.5
            },
            "extra_field": "should be ignored"
        }
        prediction_result = PredictionResult(**input_with_extra_field)
        # Ensure extra field is not present
        assert not hasattr(prediction_result, "extra_field")
        # Ensure instance is as expected
        assert prediction_result == PredictionResult(
            prediction=PredictionEnum.DEFAULT,
            probabilities=PredictedProbabilities(
                default=0.5,
                no_default=0.5
            )
        )

    # Missing field 
    @pytest.mark.unit 
    @pytest.mark.parametrize("required_fields", [
        ["prediction"], 
        ["probabilities"], 
        ["prediction", "probabilities"]
    ])
    def test_raises_validation_error_if_fields_are_missing(
            self, 
            required_fields: List[str]
    ) -> None:
        input_with_missing_fields = {
            "prediction": PredictionEnum.DEFAULT, 
            "probabilities": PredictedProbabilities(default=0.5, no_default=0.5)
        }
        for required_field in required_fields:
            del input_with_missing_fields[required_field]
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PredictionResult(**input_with_missing_fields)
        errors = exc_info.value.errors()
        # Ensure error location is the required field or fields
        assert {error["loc"][0] for error in errors} == set(required_fields)
        # Ensure error type of all errors is "missing"
        assert all(error["type"] == "missing" for error in errors)

    # None value
    @pytest.mark.unit 
    @pytest.mark.parametrize("fields", [
        ["prediction"], 
        ["probabilities"],
        ["prediction", "probabilities"]
    ])
    def test_raises_validation_error_if_fields_are_none(
            self, 
            fields: List[str]
    ) -> None:
        input_with_none = {
            "prediction": PredictionEnum.DEFAULT, 
            "probabilities": PredictedProbabilities(default=0.5, no_default=0.5)
        }
        for field in fields:
            input_with_none[field] = None
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PredictionResult(**input_with_none)
        errors = exc_info.value.errors()
        # Ensure error location is the field or fields with a None value
        assert {error["loc"][0] for error in errors} == set(fields)
        # Ensure error type of all errors is "enum" or "model_type" (which take precedence over "none_forbidden")
        assert all(error["type"] in ["enum", "model_type"] for error in errors)

    # Wrong data type (in "prediction" field) 
    @pytest.mark.unit 
    @pytest.mark.parametrize("wrong_data_type", [
        1,
        1.23,
        True,
        ["a", "list"],
        ("a", "tuple"),
        {"a": "dictionary"},
        {"a", "set"}
    ])
    def test_raises_validation_error_if_prediction_has_wrong_type(self, wrong_data_type: Any) -> None:
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PredictionResult(
                prediction=wrong_data_type,
                probabilities=PredictedProbabilities(default=0.5, no_default=0.5)
            )
        errors = exc_info.value.errors()
        # Ensure exactly one error
        assert len(errors) == 1
        # Ensure error location is the PredictionResult "prediction" field
        assert errors[0]["loc"][0] == "prediction" 
        # Ensure error type is "enum" 
        assert errors[0]["type"] == "enum" 

     # Invalid value (in "prediction" field) 
    @pytest.mark.unit 
    @pytest.mark.parametrize("invalid_enum_value", [  
        "Yes",
        "DEFAULT",     # wrong case (uppercase)
        "no default",  # wrong case (lowercase)
        "No_default",  # wrong separator and case (captical case)
    ])  
    def test_raises_validation_error_if_prediction_has_invalid_enum_value(
            self, 
            invalid_enum_value: str
    ) -> None:
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PredictionResult(
                prediction=invalid_enum_value,
                probabilities=PredictedProbabilities(default=0.5, no_default=0.5)
            )
        errors = exc_info.value.errors()
        # Ensure exactly one error
        assert len(errors) == 1
        # Ensure error location is the PredictionResult "prediction" field
        assert errors[0]["loc"][0] == "prediction" 
        # Ensure error type is "enum" 
        assert errors[0]["type"] == "enum" 

    # Wrong data type (in "probabilities" field) 
    @pytest.mark.unit 
    @pytest.mark.parametrize("wrong_data_type", [
        "a string",
        1,
        1.23,
        True,
        ["a", "list"],
        ("a", "tuple"),
        {"a", "set"}
    ])
    def test_raises_validation_error_if_probabilities_has_wrong_type(self, wrong_data_type: Any) -> None:
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PredictionResult(
                prediction=PredictionEnum.DEFAULT,
                probabilities=wrong_data_type
            )
        errors = exc_info.value.errors()
        # Ensure exactly one error
        assert len(errors) == 1
        # Ensure error location is the PredictionResult "probabilities" field
        assert errors[0]["loc"][0] == "probabilities" 
        # Ensure error type is "model_type" 
        assert errors[0]["type"] == "model_type" 

    # Missing field (within "probabilities" field) 
    @pytest.mark.unit 
    def test_raises_validation_error_if_predicted_probabilities_field_is_missing(self) -> None:
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PredictionResult(
                prediction="Default",
                probabilities={"default": 0.5}  # "no_default" missing
            )
        errors = exc_info.value.errors()
        # Ensure exactly one error
        assert len(errors) == 1
        # Ensure error location is the PredictionResult "probabilities" field > PredictedProbabilities "no_default" field
        assert errors[0]["loc"] == ("probabilities", "no_default") 
        # Ensure error type is "missing" 
        assert errors[0]["type"] == "missing" 
    
    # None value (in "probabilities" field) 
    @pytest.mark.unit 
    def test_raises_validation_error_if_predicted_probabilities_field_is_none(self) -> None:
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PredictionResult(
                prediction="Default",
                probabilities={"default": 0.5, "no_default": None} 
            )
        errors = exc_info.value.errors()
        # Ensure exactly one error
        assert len(errors) == 1
        # Ensure error location is the PredictionResult "probabilities" field > PredictedProbabilities "no_default" field
        assert errors[0]["loc"] == ("probabilities", "no_default")  
        # Ensure error type is "float_type" 
        assert errors[0]["type"] == "float_type" 

   # Wrong data type (in "probabilities" field)  
    @pytest.mark.unit 
    def test_raises_validation_error_if_predicted_probabilities_field_has_wrong_type(self) -> None:
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PredictionResult(
                prediction=PredictionEnum.DEFAULT,
                probabilities={"default": 0.5, "no_default": "wrong data type"}
            )
        errors = exc_info.value.errors()
        # Ensure exactly one error
        assert len(errors) == 1
        # Ensure error location is the PredictionResult "probabilities" field > PredictedProbabilities "no_default" field
        assert errors[0]["loc"] == ("probabilities", "no_default") 
        # Ensure error type is "float_parsing" 
        assert errors[0]["type"] == "float_parsing" 

    # Data type coersion (in "probabilities" field)
    @pytest.mark.unit 
    def test_predicted_probabilities_field_coerces_string_type_to_float(self) -> None:
        prediction_result = PredictionResult(
            prediction=PredictionEnum.DEFAULT,
            probabilities={
                "default": 0.5, 
                "no_default": "0.5"  # coerce "0.5" to 0.5
            }
        )
        expected_prediction_result = PredictionResult(
            prediction=PredictionEnum.DEFAULT,
            probabilities=PredictedProbabilities(
                default=0.5,
                no_default=0.5
            )
        )
        assert prediction_result == expected_prediction_result

    # Out-of-range value (in "probabilities" field)
    @pytest.mark.unit 
    def test_raises_validation_error_if_predicted_probabilities_value_is_out_of_range(self) -> None:
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PredictionResult(
                prediction=PredictionEnum.DEFAULT,
                probabilities={
                    "default": 0.5, 
                    "no_default": -0.001  # just below minimum of 0
                }
            )
        errors = exc_info.value.errors()
        # Ensure exactly one error
        assert len(errors) == 1
        # Ensure error location is the PredictionResult "probabilities" field > PredictedProbabilities "no_default" field
        assert errors[0]["loc"] == ("probabilities", "no_default") 
        # Ensure error type is "greater_than_equal" 
        assert errors[0]["type"] == "greater_than_equal" 

# --- Pydantic Model: PredictionResponse ---
class TestPredictionResponse:
    # Instantiation happy path
    @pytest.mark.unit
    def test_instantiation_happy_path(self) -> None:
        prediction_result_list = [
            {
                "prediction": "Default",
                "probabilities": {
                    "default": 0.8,
                    "no_default": 0.2
                } 
            },
            {
                "prediction": "No Default",
                "probabilities": {
                    "default": 0.2,
                    "no_default": 0.8
                } 
            },
        ]
        prediction_response = PredictionResponse(results=prediction_result_list)
        expected_results = [
            PredictionResult(
                prediction=PredictionEnum.DEFAULT, 
                probabilities=PredictedProbabilities(default=0.8, no_default=0.2)
            ),
            PredictionResult(
                prediction=PredictionEnum.NO_DEFAULT, 
                probabilities=PredictedProbabilities(default=0.2, no_default=0.8)
            )
        ]
        assert prediction_response.results == expected_results
        assert prediction_response.n_predictions == 2
    
    # Serialization aliases 
    @pytest.mark.unit
    def test_serialization_alias(self) -> None:
        prediction_response = PredictionResponse(
            results=[
                PredictionResult(
                    prediction=PredictionEnum.DEFAULT, 
                    probabilities=PredictedProbabilities(default=0.8, no_default=0.2)
                ),
                PredictionResult(
                    prediction=PredictionEnum.NO_DEFAULT, 
                    probabilities=PredictedProbabilities(default=0.2, no_default=0.8)
                )
            ]
        )
        output = prediction_response.model_dump(by_alias=True)
        expected_output = {
            "results": [
                {
                    "prediction": "Default",
                    "probabilities": {"Default": 0.8, "No Default": 0.2}
                },
                {
                    "prediction": "No Default",
                    "probabilities": {"Default": 0.2, "No Default": 0.8}
                }
            ],
            "n_predictions": 2
        }
        assert output == expected_output 

    # Empty "results" list
    @pytest.mark.unit
    def test_empty_results_list_is_valid(self) -> None:
        prediction_response = PredictionResponse(results=[])
        assert prediction_response.results == []
        assert prediction_response.n_predictions == 0 

    # Extra field 
    @pytest.mark.unit
    def test_extra_field_is_ignored(self) -> None:
        input_with_extra_field = {
            "results": [
                {
                    "prediction": "Default",
                    "probabilities": {"default": 0.8, "no_default": 0.2} 
                },
                {
                    "prediction": "No Default",
                    "probabilities": {"default": 0.2, "no_default": 0.8} 
                },
            ],
            "extra_field": "should be ignored"
        } 
        prediction_response = PredictionResponse(**input_with_extra_field)
        # Ensure extra field is not present
        assert not hasattr(prediction_response, "extra_field")
        # Ensure instance is as expected
        expected_prediction_response = PredictionResponse(
            results=[
                PredictionResult(
                    prediction=PredictionEnum.DEFAULT, 
                    probabilities=PredictedProbabilities(default=0.8, no_default=0.2)
                ),
                PredictionResult(
                    prediction=PredictionEnum.NO_DEFAULT, 
                    probabilities=PredictedProbabilities(default=0.2, no_default=0.8)
                )
            ],
            n_predictions=2
        )
        assert prediction_response == expected_prediction_response

    # Missing "results" field
    @pytest.mark.unit 
    def test_raises_validation_error_if_results_field_is_missing(self) -> None:
        input_with_missing_field = {}
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(**input_with_missing_field)
        errors = exc_info.value.errors()
        # Ensure exactly one error
        assert len(errors) == 1
        # Ensure error location is the PredictionResponse "results" field
        assert errors[0]["loc"][0] == "results"
        # Ensure error type is "missing"
        assert errors[0]["type"] == "missing" 

    # None value in "results" field 
    @pytest.mark.unit 
    def test_raises_validation_error_if_results_field_is_none(self) -> None:
        input_with_none = {"results": None}
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(**input_with_none)
        errors = exc_info.value.errors()
        # Ensure exactly one error
        assert len(errors) == 1
        # Ensure error location is the PredictionResponse "results" field
        assert errors[0]["loc"][0] == "results"
        # Ensure error type is "list_type" (which take precedence over "none_forbidden")
        assert errors[0]["type"] == "list_type"
 
    # Wrong data type in "results" field (must be List[PredictionResult]) 
    @pytest.mark.unit 
    @pytest.mark.parametrize("wrong_data_type", [
        "a string",
        1,
        1.23,
        True,
        ("a", "tuple"),
        {"a": "dictionary"},
        {"a", "set"}
    ])
    def test_raises_validation_error_if_results_field_has_wrong_type(self, wrong_data_type: Any) -> None:
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(results=wrong_data_type)
        errors = exc_info.value.errors()
        # Ensure exactly one error
        assert len(errors) == 1
        # Ensure error location is the PredictionResponse "results" field
        assert errors[0]["loc"][0] == "results" 
        # Ensure error type is "list_type" 
        assert errors[0]["type"] == "list_type" 

    # None value in List[PredictionResult]
    @pytest.mark.unit
    def test_raises_validation_error_if_prediction_result_list_contains_none(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(results=[
                {
                    "prediction": "Default",
                    "probabilities": {
                        "default": 0.8,
                        "no_default": 0.2
                    } 
                },
                None,  # None value in list at index 1
            ])
        errors = exc_info.value.errors()
        # Ensure exactly one error
        assert len(errors) == 1
        # Ensure error location is PredictionResponse "results" field > list index 1 
        assert errors[0]["loc"] == ("results", 1)
        # Ensure errory type is "model_type"
        assert errors[0]["type"] == "model_type"

    # Wrong data type in List[PredictionResult]
    @pytest.mark.unit
    def test_raises_validation_error_if_prediction_result_list_contains_wrong_type(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(results=[
                {
                    "prediction": "Default",
                    "probabilities": {
                        "default": 0.8,
                        "no_default": 0.2
                    } 
                },
                "a string",  # Wrong data type in list at index 1
            ])
        errors = exc_info.value.errors()
        # Ensure exactly one error
        assert len(errors) == 1
        # Ensure error location is PredictionResponse "results" field > list index 1 
        assert errors[0]["loc"] == ("results", 1)
        # Ensure errory type is "model_type"
        assert errors[0]["type"] == "model_type"

    # Missing field in PredictionResult
    def test_raises_validation_error_if_prediction_result_field_is_missing(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(results=[
                {
                    "prediction": "Default",
                    "probabilities": {
                        "default": 0.8,
                        "no_default": 0.2
                    } 
                },
                {
                    "prediction": "No Default"
                    # "probabilities" field is missing
                }            
            ])
        errors = exc_info.value.errors()
        # Ensure exactly one error
        assert len(errors) == 1
        # Ensure error location is PredictionResponse "results" field > list index 1 > PredictionResult "probabilities" field 
        assert errors[0]["loc"] == ("results", 1, "probabilities")
        # Ensure error type is "missing"
        assert errors[0]["type"] == "missing"

    # None value in a PredictionResult field
    def test_raises_validation_error_if_prediction_result_field_is_none(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(results=[
                {
                    "prediction": "Default",
                    "probabilities": {
                        "default": 0.8,
                        "no_default": 0.2
                    } 
                },
                {
                    "prediction": "No Default",
                    "probabilities": None
                }            
            ])
        errors = exc_info.value.errors()
        # Ensure exactly one error
        assert len(errors) == 1
        # Ensure error location is PredictionResponse "results" field > list index 1 > PredictionResult "probabilities" field
        assert errors[0]["loc"] == ("results", 1, "probabilities")
        # Ensure errory type is "model_type"
        assert errors[0]["type"] == "model_type"

    # Invalid value for PredictionEnum (in PredictionResult) 
    @pytest.mark.unit
    @pytest.mark.parametrize("invalid_prediction_enum", [
        "wrong string",  # the string must be an Enum member
        "DEFAULT",  # correct string but wrong casing
        "",  # empty string
        None,  
        1,  # wrong data type
        ["a", "list"],  # wrong data type
    ])
    def test_raises_validation_error_for_invalid_prediction_enum(self, invalid_prediction_enum: Any) -> None:
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(results=[
                {
                    "prediction": "Default",
                    "probabilities": {
                        "default": 0.8,
                        "no_default": 0.2
                    } 
                },
                {
                    "prediction": invalid_prediction_enum,
                    "probabilities": {
                        "default": 0.2,
                        "no_default": 0.8  
                    } 
                }            
            ])
        errors = exc_info.value.errors()
        # Ensure exactly one error
        assert len(errors) == 1
        # Ensure error location is the PredictionResponse "results" field > list index 1 > PredictionResult "prediction" field
        assert errors[0]["loc"] == ("results", 1, "prediction")
        # Ensure error type is "enum"
        assert errors[0]["type"] == "enum"

    # Wrong data type for PredictedProbabilities (in PredictionResult) 
    @pytest.mark.unit 
    def test_raises_validation_error_if_probabilities_has_wrong_type(self) -> None:
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(results=[
                {
                    "prediction": "Default",
                    "probabilities": {
                        "default": 0.8,
                        "no_default": 0.2
                    } 
                },
                {
                    "prediction": "No Default",
                    "probabilities": "a string"  # wrong data type 
                }            
            ])
        errors = exc_info.value.errors()
        # Ensure exactly one error
        assert len(errors) == 1
        # Ensure error location is PredictionResponse "results" field > list index 1 > PredictionResult "probabilities" field 
        assert errors[0]["loc"] == ("results", 1, "probabilities") 
        # Ensure error type is "model_type" (expects PredictedProbabilities model)
        assert errors[0]["type"] == "model_type" 

    # Missing field in PredictedProbabilities (in PredictionResult) 
    @pytest.mark.unit 
    def test_raises_validation_error_if_predicted_probabilities_field_is_missing(self) -> None:
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(results=[
                {
                    "prediction": "Default",
                    "probabilities": {
                        "default": 0.8,
                        "no_default": 0.2
                    } 
                },
                {
                    "prediction": "No Default",
                    "probabilities": {
                        "default": 0.2
                        # "no_default" field missing
                    }  
                }            
            ])
        errors = exc_info.value.errors()
        # Ensure exactly one error
        assert len(errors) == 1
        # Ensure error location is PredictionResponse "results" field > list index 1 > PredictionResult "probabilities" field > PredictedProbabilities "no_default" field
        assert errors[0]["loc"] == ("results", 1, "probabilities", "no_default") 
        # Ensure error type is "missing" 
        assert errors[0]["type"] == "missing" 

    # None value in a PredictedProbabilities field (in PredictionResult)
    @pytest.mark.unit 
    def test_raises_validation_error_if_predicted_probabilities_field_is_none(self) -> None:
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(results=[
                {
                    "prediction": "Default",
                    "probabilities": {
                        "default": 0.8,
                        "no_default": 0.2
                    } 
                },
                {
                    "prediction": "No Default",
                    "probabilities": {
                        "default": 0.2,
                        "no_default": None
                    }  
                }            
            ])
        errors = exc_info.value.errors()
        # Ensure exactly one error
        assert len(errors) == 1
        # Ensure error location is PredictionResponse "results" field > list index 1 > PredictionResult "probabilities" field > PredictedProbabilities "no_default" field
        assert errors[0]["loc"] == ("results", 1, "probabilities", "no_default") 
        # Ensure error type is "float_type" 
        assert errors[0]["type"] == "float_type" 

    # Wrong data type in a PredictedProbabilities field (in PredictionResult) 
    @pytest.mark.unit 
    def test_raises_validation_error_if_predicted_probabilities_field_has_wrong_type(self) -> None:
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(results=[
                {
                    "prediction": "Default",
                    "probabilities": {
                        "default": 0.8,
                        "no_default": 0.2
                    } 
                },
                {
                    "prediction": "No Default",
                    "probabilities": {
                        "default": 0.2,
                        "no_default": ["a", "list"]  # wrong data type 
                    }  
                }            
            ])
        errors = exc_info.value.errors()
        # Ensure exactly one error
        assert len(errors) == 1
        # Ensure error location is PredictionResponse "results" field > list index 1 > PredictionResult "probabilities" field > PredictedProbabilities "no_default" field
        assert errors[0]["loc"] == ("results", 1, "probabilities", "no_default") 
        # Ensure error type is "float_type" 
        assert errors[0]["type"] == "float_type" 

    # Out-of-range value in a PredictedProbabilities field (in PredictionResult)
    @pytest.mark.unit 
    def test_raises_validation_error_if_predicted_probabilities_are_out_of_range(self) -> None:
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(results=[
                {
                    "prediction": "Default",
                    "probabilities": {
                        "default": 0.8,
                        "no_default": 0.2
                    } 
                },
                {
                    "prediction": "No Default",
                    "probabilities": {
                        "default": 0.2,
                        "no_default": 1.1  # out-of-range value 
                    }  
                }            
            ])
        errors = exc_info.value.errors()
        # Ensure exactly one error
        assert len(errors) == 1
        # Ensure error location is PredictionResponse "results" field > list index 1 > PredictionResult "probabilities" field > PredictedProbabilities "no_default" field
        assert errors[0]["loc"] == ("results", 1, "probabilities", "no_default") 
        # Ensure error type is "less_than_equal" 
        assert errors[0]["type"] == "less_than_equal" 

    # Probabilities must sum to 1 error in PredictedProbabilities (in PredictionResult)
    @pytest.mark.unit 
    def test_raises_validation_error_if_predicted_probabilities_do_not_sum_to_one(self) -> None:
        # Ensure ValidationError is raised
        with pytest.raises(ValidationError) as exc_info:
            PredictionResponse(results=[
                {
                    "prediction": "Default",
                    "probabilities": {
                        "default": 0.8,
                        "no_default": 0.2
                    } 
                },
                {
                    "prediction": "No Default",
                    "probabilities": {
                        "default": 0.2,
                        "no_default": 0.9  # sum 1.1   
                    }  
                }            
            ])
        errors = exc_info.value.errors()
        # Ensure exactly one error
        assert len(errors) == 1
        # Ensure error location is PredictionResponse "results" field > list index 1 > PredictionResult "probabilities" field 
        assert errors[0]["loc"] == ("results", 1, "probabilities") 
        # Ensure error type is "value_error" 
        assert errors[0]["type"] == "value_error"
    