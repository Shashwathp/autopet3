import torch
import os
import SimpleITK as sitk
import json
from model import UNet3D
from model2 import ResUNet3D
from dataset import MedicalDataset
from torch.utils.data import DataLoader
from utils import download_weights  # Utility function to download weights

# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the input and output paths
input_ct_path = '/input/images/ct/'
input_pet_path = '/input/images/pet/'
output_path = '/output/images/automated-petct-lesion-segmentation/'
output_json_path = '/output/data-centric-model.json'

# Download the weights and set the paths
download_weights()  # This downloads the weights into /opt/algorithm/weights
model_paths = {
    'UNet3D': '/opt/algorithm/best_unet.pth',
    'ResUNet3D': '/opt/algorithm/best_resunet.pth'
}


# Initialize and load models
models = []

# UNet3D model
unet3d = UNet3D(in_channels=2, out_channels=1).to(device)
unet3d.load_state_dict(torch.load(model_paths['UNet3D'], map_location=device))
unet3d.eval()
models.append(unet3d)

# ResUNet3D model
resunet3d = ResUNet3D(in_channels=2, out_channels=1).to(device)
resunet3d.load_state_dict(torch.load(model_paths['ResUNet3D'], map_location=device))
resunet3d.eval()
models.append(resunet3d)

# Function to ensemble predictions with weights
def weighted_ensemble_predictions(models, image, weights):
    predictions = [torch.sigmoid(model(image)).detach() for model in models]
    weighted_predictions = sum(w * pred for w, pred in zip(weights, predictions))
    return weighted_predictions

# Set weights for ensembling (0.4 for UNet3D, 0.6 for ResUNet3D)
weights = [0.4, 0.6]

# Load the input files (CT and PET)
ct_uuid = os.listdir(input_ct_path)[0].split('.')[0]  # Extract the UUID
ct_image = sitk.ReadImage(os.path.join(input_ct_path, f"{ct_uuid}.mha"))
pet_image = sitk.ReadImage(os.path.join(input_pet_path, f"{ct_uuid}.mha"))

# Convert to numpy arrays and stack them as input
ct_array = sitk.GetArrayFromImage(ct_image)
pet_array = sitk.GetArrayFromImage(pet_image)
input_image = torch.tensor([ct_array, pet_array], dtype=torch.float32).unsqueeze(0).to(device)

# Get ensembled predictions
ensembled_output = weighted_ensemble_predictions(models, input_image, weights)

# Threshold the output to get binary mask
binary_mask = (ensembled_output > 0.5).float()

# Convert back to SimpleITK image format
binary_mask_image = sitk.GetImageFromArray(binary_mask.cpu().numpy().squeeze())
binary_mask_image.CopyInformation(ct_image)

# Save the output segmentation
output_segmentation_path = os.path.join(output_path, f"{ct_uuid}.mha")
sitk.WriteImage(binary_mask_image, output_segmentation_path)

# Create data-centric-model.json (set to False for now)
data_centric_json = {"datacentric": False}
with open(output_json_path, 'w') as json_file:
    json.dump(data_centric_json, json_file)
