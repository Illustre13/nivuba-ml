# Imports
import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC 
from selenium.webdriver.remote.webdriver import WebDriver


# --- Helper Functions ---
# Enter a number in a Gradio Number input
def set_number_input(webdriver: WebDriver, number_input: str, value: int) -> None:
    number_input_element = WebDriverWait(webdriver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, f"input[aria-label='{number_input}']")))
    number_input_element.send_keys(value)

# Enter a number in a Gradio Slider input
def set_slider_input(webdriver: WebDriver, slider_input: str, value: int) -> None:
    # Sliders have 2 input fields, a number input and a range slider, use the number input (identified via aria-label)
    slider_input_element = WebDriverWait(webdriver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, f"input[aria-label='number input for {slider_input}']")))
    slider_input_element.clear()
    slider_input_element.send_keys(value)

# Select an option from a Gradio Dropdown input
def set_dropdown_input(webdriver: WebDriver, dropdown_input: str, value: str) -> None:
    # First click Dropdown to bring up the options, then click on an option 
    dropdown_input_element = WebDriverWait(webdriver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, f"input[aria-label='{dropdown_input}']")))
    dropdown_input_element.click()
    dropdown_menu_option = WebDriverWait(webdriver, 10).until(EC.element_to_be_clickable((By.XPATH, f"//ul[contains(@class, 'options')]//li[text()='{value}']")))
    assert dropdown_menu_option.text == value
    dropdown_menu_option.click()

# Extract prediction text from Gradio Textbox output rendered in a <textarea>
def extract_prediction_text(webdriver: WebDriver) -> str:
    prediction_text_element = WebDriverWait(webdriver, 10).until(EC.presence_of_element_located((By.XPATH, "//div[@id='prediction-text']//textarea")))
    prediction_text = prediction_text_element.get_attribute("value")
    return prediction_text


# --- Fixtures ---
# Create a Chrome webdriver with custom options
@pytest.fixture
def driver() -> WebDriver:
    # Customize options for a Chrome webdriver
    chrome_options = Options()
    # Disable Chrome sandbox to prevent Chrome crashes due to restricted security setup 
    chrome_options.add_argument("--no-sandbox")  
    # Disable Chrome shared memory (uses temporary storage instead) to prevent crashes due to limited ressources in Docker containers
    chrome_options.add_argument("--disable-dev-shm-usage")  
    # Run Chrome without opening a browser window
    # chrome_options.add_argument("--headless")  
    # Create a Chrome webdriver with custom options 
    driver = webdriver.Chrome(options=chrome_options)    
    yield driver
    # Close Chrome webdriver to free up memory and other system resources
    driver.quit()


# End-to-end happy path test that simulates a user submitting the form and receiving a prediction in the frontend UI 
@pytest.mark.e2e
def test_happy_path(driver: WebDriver) -> None:
    # Get request to frontend Gradio UI  
    # Make sure the Docker container is running locally and port 7860 is mapped
    driver.get("http://localhost:7860")

    # Set inputs in Gradio UI
    set_number_input(driver, "Age", 30)
    set_dropdown_input(driver, "Married/Single", "Single")
    set_number_input(driver, "Income", 300000)
    set_dropdown_input(driver, "Car Ownership", "No")
    set_dropdown_input(driver, "House Ownership", "Neither Rented Nor Owned")
    set_slider_input(driver, "Current House Years", 11)
    set_dropdown_input(driver, "City", "Sikar")
    set_dropdown_input(driver, "State", "Rajasthan")
    set_dropdown_input(driver, "Profession", "Artist")
    set_slider_input(driver, "Experience", 3)
    set_slider_input(driver, "Current Job Years", 3)

    # Click predict button
    predict_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "predict-button")))
    predict_button.click()

    # Extract predicted probabilities
    default_probability_element = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.XPATH, "//div[@id='pred-proba-label']//dt[text()='Default']/following-sibling::dd")))
    no_default_probability_element = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.XPATH, "//div[@id='pred-proba-label']//dt[text()='No Default']/following-sibling::dd")))
    default_probability = int(default_probability_element.text.replace("%", ""))
    no_default_probability = int(no_default_probability_element.text.replace("%", ""))
    # Extract prediction text 
    prediction_text = extract_prediction_text(driver)
    
    # Ensure prediction text is as expected
    assert prediction_text in ["Default", "No Default"]
    # Ensure probabilities are numbers between 0 and 100
    assert 0 <= default_probability <=100
    assert 0 <= no_default_probability <= 100
    # Ensure probabilities sum to approximately 100
    sum = default_probability + no_default_probability
    assert 99 <= sum <= 101  # allow for rounding edge cases


# End-to-end happy path test that simulates a user submitting the form with missing optional fields and receiving a prediction in the frontend UI 
@pytest.mark.e2e
def test_happy_path_with_missing_optional_fields(driver: WebDriver) -> None:
    # Get request to frontend Gradio UI  
    # Make sure the Docker container is running locally and port 7860 is mapped
    driver.get("http://localhost:7860")

    # Set inputs in Gradio UI
    set_number_input(driver, "Age", 30)
    # Married/Single input (optional) is missing
    set_number_input(driver, "Income", 300000)
    # Car Ownership input (optional) is missing
    # House Ownership input (optional) is missing
    set_slider_input(driver, "Current House Years", 11)
    set_dropdown_input(driver, "City", "Sikar")
    set_dropdown_input(driver, "State", "Rajasthan")
    set_dropdown_input(driver, "Profession", "Artist")
    set_slider_input(driver, "Experience", 3)
    set_slider_input(driver, "Current Job Years", 3)

    # Click predict button
    predict_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "predict-button")))
    predict_button.click()

    # Extract predicted probabilities
    default_probability_element = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.XPATH, "//div[@id='pred-proba-label']//dt[text()='Default']/following-sibling::dd")))
    no_default_probability_element = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.XPATH, "//div[@id='pred-proba-label']//dt[text()='No Default']/following-sibling::dd")))
    default_probability = int(default_probability_element.text.replace("%", ""))
    no_default_probability = int(no_default_probability_element.text.replace("%", ""))
    # Extract prediction text 
    prediction_text = extract_prediction_text(driver)
    
    # Ensure prediction text is as expected
    assert prediction_text in ["Default", "No Default"]
    # Ensure probabilities are numbers between 0 and 100
    assert 0 <= default_probability <=100
    assert 0 <= no_default_probability <= 100
    # Ensure probabilities sum to approximately 100
    sum = default_probability + no_default_probability
    assert 99 <= sum <= 101  # allow for rounding edge cases


# End-to-end test that simulates a user submitting the form with missing required fields and receiving an error message in the frontend UI 
@pytest.mark.e2e
def test_error_message_for_empty_required_fields(driver: WebDriver) -> None:
    # Get request to frontend Gradio UI  
    # Make sure the Docker container is running locally and port 7860 is mapped
    driver.get("http://localhost:7860")

    # Set inputs in Gradio UI
    set_number_input(driver, "Age", 30)
    set_dropdown_input(driver, "Married/Single", "Single")
    set_number_input(driver, "Income", 300000)
    set_dropdown_input(driver, "Car Ownership", "No")
    set_dropdown_input(driver, "House Ownership", "Neither Rented Nor Owned")
    set_slider_input(driver, "Current House Years", 11)
    # City input (required) is missing
    # State input (required) is missing
    # Profession input (required) is missing
    set_slider_input(driver, "Experience", 3)
    set_slider_input(driver, "Current Job Years", 3)

    # Click predict button
    predict_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "predict-button")))
    predict_button.click()

    # Extract error message in predicted probabilities element (should be an empty str)
    # Note: Error path renders str in single h2 element whereas success path renders dict in multiple dl/dd elements
    probabilities_element = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.XPATH, "//div[@id='pred-proba-label']//h2"))) 
    error_msg_in_probabilities = probabilities_element.text 

    # Extract error message in prediction text
    error_msg_in_prediction = extract_prediction_text(driver)
    
    # Ensure error message is as expected
    expected_error_msg = (
        "Input Error! Please check your inputs and try again.\n"
        "City: Select a city from the list.\n"
        "State: Select a state from the list.\n"
        "Profession: Select a profession from the list.\n"
    )
    assert error_msg_in_prediction == expected_error_msg
    assert error_msg_in_probabilities == ""


# End-to-end test that simulates a user submitting out-of-range values and receiving an error message in the frontend UI 
@pytest.mark.e2e
def test_error_message_for_out_of_range_values(driver: WebDriver) -> None:
    # Get request to frontend Gradio UI  
    # Make sure the Docker container is running locally and port 7860 is mapped
    driver.get("http://localhost:7860")

    # Set inputs in Gradio UI
    set_number_input(driver, "Age", 5)  # out-of-range
    set_dropdown_input(driver, "Married/Single", "Single")
    set_number_input(driver, "Income", -500)  # out-of-range
    set_dropdown_input(driver, "Car Ownership", "No")
    set_dropdown_input(driver, "House Ownership", "Neither Rented Nor Owned")
    set_slider_input(driver, "Current House Years", 11)
    set_dropdown_input(driver, "City", "Sikar")
    set_dropdown_input(driver, "State", "Rajasthan")
    set_dropdown_input(driver, "Profession", "Artist")
    set_slider_input(driver, "Experience", 3)
    set_slider_input(driver, "Current Job Years", 3)

    # Click predict button
    predict_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "predict-button")))
    predict_button.click()

    # Extract error message in predicted probabilities element (should be an empty str)
    # Note: Error path renders str in single h2 element whereas success path renders dict in multiple dl/dd elements
    probabilities_element = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.XPATH, "//div[@id='pred-proba-label']//h2"))) 
    error_msg_in_probabilities = probabilities_element.text 

    # Extract error message in prediction text
    error_msg_in_prediction = extract_prediction_text(driver)

    # Ensure error message is as expected
    expected_error_msg = (
        "Input Error! Please check your inputs and try again.\n"
        "Age: Enter a number between 21 and 79.\n"
        "Income: Enter a number that is 0 or greater.\n"
    )
    assert error_msg_in_prediction == expected_error_msg
    assert error_msg_in_probabilities == ""


# End-to-end test that simulates a user submitting the form without providing any inputs and receiving an error message 
@pytest.mark.e2e
def test_error_message_for_no_user_inputs(driver: WebDriver) -> None:
    # Get request to frontend Gradio UI  
    # Make sure the Docker container is running locally and port 7860 is mapped
    driver.get("http://localhost:7860")

    # No user inputs, just click predict button
    predict_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.ID, "predict-button")))
    predict_button.click()

    # Extract error message in predicted probabilities element (should be an empty str)
    # Note: Error path renders str in single h2 element whereas success path renders dict in multiple dl/dd elements
    probabilities_element = WebDriverWait(driver, 30).until(EC.presence_of_element_located((By.XPATH, "//div[@id='pred-proba-label']//h2"))) 
    error_msg_in_probabilities = probabilities_element.text 

    # Extract error message in prediction text
    error_msg_in_prediction = extract_prediction_text(driver)

    # Ensure error message is as expected
    expected_error_msg = (
        "Input Error! Please check your inputs and try again.\n"
        "Age: Enter a number between 21 and 79.\n"
        "Income: Enter a number that is 0 or greater.\n"
        "City: Select a city from the list.\n"
        "State: Select a state from the list.\n"
        "Profession: Select a profession from the list.\n"
        # Current House Years, Experience and Current Job Years are also required fields, but 
        # the Gradio Slider inputs provide default values that can't be changed to empty fields
    )
    assert error_msg_in_prediction == expected_error_msg
    assert error_msg_in_probabilities == ""
