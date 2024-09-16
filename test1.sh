#!/bin/bash

# Ensure model weights are downloaded
echo "Checking and downloading model weights..."
bash download_model_weights.sh

# Test command for running the model
# Modify the commands based on how you are running your tests
echo "Running tests..."

# Test with ResUNet3D model
python3 process.py --model ResUNet3D --weights /workspace/weights/best_resunet.pth

# Test with UNet3D model
python3 process.py --model UNet3D --weights /workspace/weights/best_unet.pth

echo "Tests completed!"
