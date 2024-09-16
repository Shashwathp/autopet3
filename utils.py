import os
import subprocess

def download_weights():
    """
    Downloads model weights using the download_model_weights.sh script if they don't already exist.
    """
    weights_path = '/opt/algorithm/best_unet.pth'  # You can add paths for other models similarly
    if not os.path.exists(weights_path):
        script_path = '/opt/algorithm/download_model_weights.sh'
        if os.path.exists(script_path):
            subprocess.run(['bash', script_path], check=True)
        else:
            raise FileNotFoundError("Weights download script not found.")
    else:
        print("Weights already downloaded. Skipping download.")