#!/bin/bash

# Define the download directory
download_dir="/opt/algorithm"

# Create the download directory if it doesn't exist
mkdir -p "${download_dir}"

# Download best_resunet.pth from Dropbox
echo "Downloading best_resunet.pth..."
wget "https://www.dropbox.com/scl/fi/seoac75f5tmh0rqp4r12n/best_resunet.pth?rlkey=ksyk9sojgdmm7708eab2bjob5" -O "${download_dir}/best_resunet.pth"

# Download best_unet.pth from Dropbox
echo "Downloading best_unet.pth..."
wget "https://www.dropbox.com/scl/fi/tfmc1g74hr0ddlzqhqvzd/best_unet.pth?rlkey=4133092xc9n8xftjwt6vpiw9u" -O "${download_dir}/best_unet.pth"

echo "Download completed!"
