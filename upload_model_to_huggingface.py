import os
from huggingface_hub import HfApi, HfFolder

def upload_model_to_huggingface(local_model_path, repo_id):
    """
    Uploads a local model to a specified Hugging Face repository.

    Args:
    - local_model_path (str): Path to the local model directory.
    - repo_id (str): Hugging Face repository ID (e.g., 'kmcowan/rockymtn_gpt2').

    Returns:
    - None
    """
    # Ensure the local model path exists
    if not os.path.exists(local_model_path):
        raise FileNotFoundError(f"Local model path '{local_model_path}' does not exist.")

    # Ensure the Hugging Face token is available
    token = "[your huggingface tokey]" #HfFolder.get_token()
    if token is None:
        raise ValueError("Hugging Face token not found. Please log in using `huggingface-cli login`.")

    # Initialize the Hugging Face API
    api = HfApi()

    # Upload the local model to the specified repository
    print(f"Uploading model from '{local_model_path}' to Hugging Face repository '{repo_id}'...")
    api.upload_folder(
        folder_path=local_model_path,
        repo_id=repo_id,
        repo_type="model"
    )
    print("Model uploaded successfully.")

if __name__ == "__main__":
    # Define the local model path and Hugging Face repository ID
    local_model_path = "huggingface_updated"  # Replace with your local model directory
    repo_id = "kmcowan/rockymtn_gpt2"  # Replace with your Hugging Face repository ID

    # Upload the model to Hugging Face
    upload_model_to_huggingface(local_model_path, repo_id)