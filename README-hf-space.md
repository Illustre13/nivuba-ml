---
title: Loan Default Prediction
subtitle: Submit customer application data to predict loan default
emoji: 💰
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
models:
  - JensBender/loan-default-prediction-pipeline
tags:
  - finance
  - credit-risk
  - loan-default
  - tabular-data
  - scikit-learn
  - random-forest
  - gradio
  - fastapi
  - docker
---

## 🏦 Loan Default Prediction App
A web application that predicts loan default based on customer application data, helping financial institutions make data-driven lending decisions.  
Built with `Gradio`, `FastAPI`, and a `scikit-learn` Random Forest model trained on over 250,000 loan applications.

### How to Use 
1.  **Fill in Form**: Enter applicant details such as age, income, and experience.
2.  **Click Predict**: The app will process your input and return a "Default" or "No Default" prediction along with probabilities.
3.  **Interpret Responsibly**: Use the prediction to support decision making, do **not** use for fully automated decisions without human oversight.  

### Use via API
You can also send requests directly to the FastAPI backend for programmatic access. This is useful for integrating the model into other applications or systems.

Example API usage with Python's `requests` library:
```python
import requests 

# Create example applicant data (JSON payload)
applicant_data = {
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
}

# API request to FastAPI predict endpoint on Hugging Face Spaces
prediction_api_url = "https://jensbender-loan-default-prediction-app.hf.space/api/predict"
response = requests.post(prediction_api_url, json=applicant_data)

# Check if request was successful
response.raise_for_status()

# Extract prediction and probability of default
prediction_response = response.json()
prediction_result = prediction_response["results"][0]
prediction = prediction_result["prediction"]
default_probability = prediction_result["probabilities"]["Default"]

# Show results
print(f"Probability of default: {default_probability * 100:.1f}% (threshold: 29.0%)")
print(f"Prediction: {prediction}")
```

### How It Works
1. **Gradio Frontend (UI Layer)**  
    - Provides a clean and simple form for data entry.  
    - Sends form data as JSON to the backend API.  
    - Displays prediction results and probabilities in real time.
2. **FastAPI Backend (API Layer)**  
    - Receives requests from the `Gradio` frontend or direct API requests.  
    - Loads the pre-trained pipeline from the [Hugging Face Hub](https://huggingface.co/JensBender/loan-default-prediction-pipeline).  
    - Validates and passes data through the pipeline, and applies the decision threshold.  
    - Returns JSON responses containing predictions and probabilities.
3. **ML Pipeline (Model Layer)**  
   - Implements a `scikit-learn` pipeline with a Random Forest Classifier model and preprocessing.  
   - Performs feature engineering, scaling, and encoding.  
   - Outputs predicted probabilities for both classes ("Default" and "No Default").
4. **Deployment Environment**
    - Packaged as a single `Docker` container.  
    - Runs seamlessly on Hugging Face Spaces using the Docker SDK.  

### Model Performance
The Random Forest model achieved an **AUC-PR of 0.59** on the test set. The most influential features are income, age, and state default rate (derived via feature engineering).

**Classification Report (Test)**
|                        | Precision | Recall | F1-Score | Samples |
|:-----------------------|:----------|:-------|:---------|:--------|
| Class 0: Non-Defaulter | 0.97      | 0.90   | 0.93     | 22,122  |
| Class 1: Defaulter     | 0.51      | 0.79   | 0.62     | 3,078   |
| Accuracy               |           |        | 0.88     | 25,200  |
| Macro Avg              | 0.74      | 0.84   | 0.78     | 25,200  |
| Weighted Avg           | 0.91      | 0.88   | 0.89     | 25,200  |

### Resources
| Component | Description | Link |
|------------|--------------|------|
| **Source Code** | Full project repository with training, evaluation, and deployment scripts | [GitHub](https://github.com/JensBender/loan-default-prediction) |
| **Model Pipeline** | Pre-trained `scikit-learn` pipeline with Random Forest Classifier and preprocessing | [Hugging Face Hub](https://huggingface.co/JensBender/loan-default-prediction-pipeline) |
| **Web App** | Live, interactive demo with Gradio frontend and FastAPI backend | [Hugging Face Spaces](https://huggingface.co/spaces/JensBender/loan-default-prediction-app) |

### Responsible Use
The model and by extension this web app and API are intended to be used as a tool to support credit risk assessment. They can be integrated into decision-making workflows to provide a quantitative measure of default risk for loan applicants.

This model is **not** intended for:
- Fully automated lending decisions without human oversight. The model's predictions should not be the sole factor in any financial decision.
- Evaluating applicants from demographic, geographic, or socioeconomic backgrounds not represented in the training data.
- Use in a production environment without rigorous, ongoing validation and fairness audits. 

### Bias, Risks, and Limitations  
The model was trained on historical data that may carry biases related to socioeconomic status, geography, or other demographic factors, potentially leading to unfair predictions for certain groups. The model can be overconfident on misclassified edge cases, assigning high probabilities to incorrect predictions. Confidence scores should not be relied upon without additional scrutiny.

**Recommendations**  
- **Human in the Loop:** Always use this model as part of a broader decision-making framework that includes human oversight.
- **Fairness and Bias Audits:** Before deploying this model in a production environment, conduct thorough fairness and bias analyses to ensure it performs equally across different demographic groups.
- **Model Monitoring:** Continuously monitor the model's performance and predictions to detect and mitigate any performance degradation or emerging biases.

### License
The source code for this web app on Hugging Face Spaces and the source code of the overall project on [GitHub](https://github.com/JensBender/loan-default-prediction) is licensed under the [MIT License](LICENSE). The model pipeline is licensed under [Apache-2.0](https://huggingface.co/JensBender/loan-default-prediction-pipeline/resolve/main/LICENSE).

### Citation
If you use this model or app in your work, please cite it as follows:
```bibtex
@misc{bender_loan_default_prediction_2025,
  author       = {Bender, Jens},
  title        = {Loan Default Prediction Pipeline},
  year         = {2025},
  publisher    = {Hugging Face},
  url          = {https://huggingface.co/JensBender/loan-default-prediction-pipeline},
  note         = {Version 1.0. A scikit-learn Random Forest pipeline for predicting loan defaults. Trained on 252,000 loan applications. Source code available at \url{https://github.com/JensBender/loan-default-prediction}. Licensed under Apache-2.0.}
}
```
