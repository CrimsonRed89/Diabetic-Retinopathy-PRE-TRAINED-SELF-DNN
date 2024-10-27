# Diabetic Retinopathy Detection using Deep Learning
This project aims to detect diabetic retinopathy from retinal images using deep learning models. We experimented with various pretrained CNN architectures through transfer learning and developed two custom models, DR Net V1 and DR Net V2. The objective is to identify the presence and severity of diabetic retinopathy to aid in early diagnosis and treatment.

# Project Overview
Diabetic Retinopathy is a severe eye condition that affects individuals with diabetes. Early detection is critical to prevent vision loss. This project employs deep learning techniques to classify retinal images into different categories of diabetic retinopathy severity.

# Links to Kaggle Notebooks :
## PRETRAINED_TFLEARNING_MODELS : 
https://www.kaggle.com/code/aayuzzzz/dr-new
## SELF CNN & FASTAI DRNET : 
https://www.kaggle.com/code/aayuzzzz/self-dr-best
 
# Key Features
Pretrained Models Used: VGG19, VGG16, ResNet 101, ResNet 50, EfficientNet, MobileNet, and Inception.

# Custom Models:
## DR Net V1: 
A custom-built CNN architecture tailored for retinal image classification.
## DR Net V2: 
An advanced version of DR Net V1 using FastAI, optimized for better performance and efficiency.

# Data Preprocessing: The images were preprocessed for feature extraction, skipping the traditional train/test folder structure.
Feature Extraction Techniques: Local Binary Pattern (LBP), Local Directional Pattern (LDP), Discrete Wavelet Transform (DWT), Hough Transform, Gabor Transform, and Discrete Fourier Transform (DFT).
Project Structure.

# Data Preprocessing: 
Images were normalized and resized to 224x224 pixels. Advanced data augmentation techniques were applied to enhance model robustness.

# Model Training: 
Each model was trained using the preprocessed images. Transfer learning was employed for the pretrained models, with the final layers fine-tuned to adapt to our specific dataset.

# Evaluation: 
Models were evaluated on various metrics, though the specific metrics are not included in this README. Models showed varying degrees of accuracy, highlighting areas for improvement before deployment.

# Feature Extraction Visualization: 
Feature extraction images for various techniques (e.g., LBP, LDP) were generated to visualize the model's focus areas and feature learning.

# Models
## 1. Pretrained Models
VGG16 and VGG19: Known for their depth and simplicity, used as baselines for comparison.
ResNet 50 and ResNet 101: Leveraging residual connections to combat the vanishing gradient problem.
EfficientNet: A scalable model balancing accuracy and efficiency.
MobileNet: Optimized for mobile and edge devices with a lightweight architecture.
Inception: Employs multiple convolution filter sizes to capture diverse features.

## 2. Custom Models
DR Net V1: A custom CNN with layers optimized for diabetic retinopathy detection, focusing on extracting detailed retinal features.
DR Net V2: Built on FastAI, incorporating advanced techniques for improved learning and generalization.

# How to Run
## Clone the Repository:
git clone https://github.com/CrimsonRed89/Diabetic-Retinopathy-PRE-TRAINED-SELF-DNN.git

cd diabetic-retinopathy-detection

## Install Dependencies: 
Make sure to install all necessary packages

## Data Preparation: 
Ensure the data is correctly formatted and placed in the designated folder.

## Training the Models: 
Train any of the models by running

# Results and Future Work
The models achieved varied accuracy levels, and some need further tuning and optimization. There is a recognized need for improvement in handling class imbalance and enhancing the precision of certain classes.

# Future Steps:
## Model Refinement: 
Continue refining the custom models, focusing on optimizing layer configurations and hyperparameters.
## Deployment: 
Deploy the best-performing model into a production environment for real-world application.
## Further Exploration: 
Explore ensemble methods to combine the strengths of different models.

# Contributing
Contributions are welcome! Please submit a pull request or open an issue for suggestions and improvements.

# Contact
For any queries or discussions, please reach out at aayusharora0304@gmail.com
