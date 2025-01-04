# A Class-Weighted Ensemble Approach for Robust Multitracer Lesion Segmentation in Whole-Body PET/CT Scans  
**AutoPET Challenge, MICCAI 2024**  

This project presents a class-weighted ensemble approach for automated lesion segmentation in whole-body PET/CT scans. The methodology is developed as part of the MICCAI 2024 autoPET III Challenge, focusing on advancing multitracer, multicenter generalization in medical imaging.

## Overview  
Building upon the foundation of the autoPET Challenges, this work addresses the critical challenge of segmenting tumor lesions in PET/CT scans using diverse tracers such as Fluorodeoxyglucose (FDG) and Prostate-Specific Membrane Antigen (PSMA) from multiple clinical centers. Manual lesion segmentation remains a bottleneck in clinical workflows due to its labor-intensive nature. This project aims to overcome these challenges using robust and generalizable machine-learning techniques.

## Approach  
- **Ensemble Design**: Developed a novel class-weighted ensemble of three architectures: UNET, ResUNET, and TransUNET. This combination captures global and fine-grained features, optimizing lesion segmentation.  
- **Multi-scale Detection**: Leveraged architectural diversity to address lesions of varying sizes and complexity, ensuring robustness across tracers and imaging protocols.  
- **Performance**: Achieved a Dice coefficient of 0.81 in lesion segmentation, demonstrating strong generalization capabilities across the heterogeneous dataset provided by the autoPET III Challenge.  

## Results  
![Example of PET/CT Scan Results](images/pet_ct_example.png "PET/CT Example")

## How to Add Images to README.md  
To include an image in your GitHub README, use the following syntax:  

```markdown
![Alt Text](image_path_or_URL "Optional Title")
