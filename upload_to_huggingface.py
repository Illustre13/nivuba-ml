from huggingface_hub import HfApi

# --- Constants --- 
# Select what to upload
UPLOAD_PIPELINE = False
UPLOAD_MODEL_CARD = True
UPLOAD_IMAGES = False

# Hugging Face Hub repository (make sure to create it first)
HF_HUB_REPO = "JensBender/loan-default-prediction-pipeline"  

# Pipeline
LOCAL_PIPELINE_PATH = "models/loan_default_rf_pipeline.joblib"  
HF_PIPELINE_PATH = "loan_default_rf_pipeline.joblib"
TAG = "v1.0"  # version tag

# Model card
LOCAL_MODEL_CARD_PATH = "README-hf-hub.md"
HF_MODEL_CARD_PATH = "README.md"

# --- Hugging Face API client --- 
api = HfApi()

# --- Upload Pipeline ---
if UPLOAD_PIPELINE:
    print(f"Uploading '{HF_PIPELINE_PATH}' to Hugging Face Hub repository '{HF_HUB_REPO}'...")
    api.upload_file(
        path_or_fileobj=LOCAL_PIPELINE_PATH,
        repo_id=HF_HUB_REPO,
        path_in_repo=HF_PIPELINE_PATH,
        repo_type="model",
        commit_message=f"Update '{HF_PIPELINE_PATH}' on Hugging Face Hub"
    )
    print("Successfully uploaded pipeline.")

    # Add the version tag to the latest commit
    print(f"Creating tag '{TAG}'...")
    api.create_tag(
        repo_id=HF_HUB_REPO,
        tag=TAG,
        repo_type="model",
        exist_ok=True  # prevents error if the tag already exists
    )
    print(f"Successfully created tag '{TAG}' in repo '{HF_HUB_REPO}'.")

# --- Upload Model Card ---
if UPLOAD_MODEL_CARD:
    print(f"Uploading model card '{LOCAL_MODEL_CARD_PATH}' renamed as '{HF_MODEL_CARD_PATH}' to Hugging Face Hub repository '{HF_HUB_REPO}'...")
    api.upload_file(
        path_or_fileobj=LOCAL_MODEL_CARD_PATH,
        repo_id=HF_HUB_REPO,
        path_in_repo=HF_MODEL_CARD_PATH,
        repo_type="model",
        commit_message="Update model card 'README.md' on Hugging Face Hub"
    )
    print("Successfully uploaded model card.")

# --- Upload Images ---
if UPLOAD_IMAGES:
    print(f"Uploading images to Hugging Face Hub repository '{HF_HUB_REPO}'...")
    images_to_upload = ["header-image.webp", "rf_confusion_matrix_test.png", "rf_feature_importance_final.png"]
    api.upload_folder(
        folder_path="images",
        repo_id=HF_HUB_REPO,
        path_in_repo="images",
        allow_patterns=images_to_upload,  # only upload a subset of images
        repo_type="model", 
        commit_message="Upload images"
    )
    print("Successfully uploaded images.")
