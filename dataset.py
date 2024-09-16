import os
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom

class MedicalDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, target_size=(128, 128, 128),num_labels=100):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.target_size = target_size
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('_0000.nii.gz')])[:num_labels]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        ct_file = os.path.join(self.images_dir, self.image_files[idx])
        pet_file = ct_file.replace('_0000.nii.gz', '_0001.nii.gz')
        label_file = os.path.join(self.labels_dir, self.image_files[idx].replace('_0000.nii.gz', '.nii.gz'))

        # Load NIfTI images
        ct_image = nib.load(ct_file).get_fdata()
        pet_image = nib.load(pet_file).get_fdata()
        label = nib.load(label_file).get_fdata()

        # Normalize images (min-max normalization)
        ct_image = (ct_image - np.min(ct_image)) / (np.max(ct_image) - np.min(ct_image))
        pet_image = (pet_image - np.min(pet_image)) / (np.max(pet_image) - np.min(pet_image))

        # Resize images to the target size (128, 128, 128)
        ct_image = self.resize_image(ct_image, self.target_size)
        pet_image = self.resize_image(pet_image, self.target_size)
        label = self.resize_image(label, self.target_size, order=0)  # Nearest-neighbor interpolation for labels

        # Stack the PET and CT scans
        image = np.stack([ct_image, pet_image], axis=0)  # shape (2, D, H, W)

        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        # Convert to torch tensors
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return image, label

    def resize_image(self, image, target_size, order=3):
        """Resize the image to the target size using interpolation."""
        current_size = image.shape
        zoom_factors = [target_size[i] / current_size[i] for i in range(3)]
        return zoom(image, zoom_factors, order=order)