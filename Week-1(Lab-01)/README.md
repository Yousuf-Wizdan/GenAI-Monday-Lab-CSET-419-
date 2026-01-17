# Week-1 (Lab-01): Dog Breed Dataset Generation & Classification Model Training

## Overview
This lab focuses on creating a custom dog breed dataset using AI-generated images and training a CNN classifier to recognize different dog breeds.

## Objectives
- Generate synthetic dog images using Stable Diffusion
- Create a dataset of 20 dog breeds (20 images each = 400 total images)
- Train a ResNet-18 model for dog breed classification
- Evaluate model performance on train/validation/test splits

## Dataset
**20 Dog Breeds:**
- Golden Retriever, German Shepherd, Labrador Retriever, Beagle, Pug
- Rottweiler, Doberman Pinscher, Siberian Husky, French Bulldog, Border Collie
- Chihuahua, Pomeranian, Great Dane, Shih Tzu, Boxer
- Dachshund, Akita, Cocker Spaniel, Bernese Mountain Dog, Australian Shepherd

**Dataset Structure:**
- Total: 400 images (20 breeds × 20 images each)
- Split: 70% Train / 15% Validation / 15% Test

## Workflow

### Step 1: Image Generation
- Uses Stable Diffusion v1.5 (`runwayml/stable-diffusion-v1-5`)
- Generates high-quality, realistic dog photos with prompts
- Organized in breed-specific folders

### Step 2: Data Preparation
- Image transformations (resize, augmentation, normalization)
- Dataset split: 70% Train / 15% Validation / 15% Test
- DataLoaders with batch size 16

### Step 3: Model Training
- **Architecture:** ResNet-18 (pretrained, fine-tuned)
- **Input:** 224×224 RGB images
- **Output:** 20-class classification
- **Loss Function:** CrossEntropyLoss
- **Optimizer:** Adam (lr=0.0003)
- **Epochs:** 10
- Tracks training loss, train accuracy, and validation accuracy per epoch

### Step 4: Model Evaluation
- Final test set evaluation
- Reports test accuracy on unseen data

## Files
- `GenAI_LAB_01.ipynb` - Jupyter notebook with full implementation
- `genai_lab_01.py` - Python script version
- `dog_dataset/` - Generated images organized by breed

## Requirements
```
diffusers
transformers
accelerate
safetensors
torch
torchvision
```

## Usage
Run the notebook in Google Colab with GPU runtime for best performance. Make sure to mount Google Drive for dataset storage.
