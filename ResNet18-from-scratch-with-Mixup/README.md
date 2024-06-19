# ResNet-18 Training from Scratch on ImageNet with Mixup Data Augmentation

## Context

This project aims to train a ResNet-18 model from scratch on the ImageNet dataset, using the Mixup approach for data augmentation. Mixup is a simple yet effective data augmentation technique that improves model generalization by creating new training examples through linear interpolations of random pairs of examples and their labels. This project explores the advantages of Mixup in enhancing the performance of deep learning models on large-scale datasets.

## Project Overview

In this project, we build and train a ResNet-18 model from scratch using the ImageNet dataset. The primary objective is to implement Mixup data augmentation from scratch and integrate it with standard data augmentation techniques used in the original ResNet paper. The training process leverages GPU acceleration to handle the computational demands of training on the large ImageNet dataset.

## Objectives

- Build a ResNet-18 architecture from scratch.
- Implement Mixup data augmentation.
- Integrate Mixup with standard data augmentation techniques.
- Train the ResNet-18 model on the ImageNet dataset using GPU.
- Evaluate the performance of the model with and without Mixup.

## Dataset

The ImageNet dataset contains 1.3 million training images (1,300 images per classes) and 50,000 validation images, spanning 1,000 classes. Each image is labeled with a single class. For this project, we use the ImageNet Object Localization Challenge subset available on Kaggle.

## Repository Structure

- `train.py`: Contains the training loop, logic for training the ResNet-18 model and mixup configuration.
- `utils.py`: Includes utility functions such as display_images and count_parameters.
- `dataset.py`: Handles data loading, preprocessing, and augmentation for the ImageNet dataset.
- `args.py`: Contains global variables
- `main.py`: The main script to set up the training process, including model initialization and configuration.
- `model.py`: Defines the ResNet-18 architecture.

## How to Run the Project

### Prerequisites

- Python 3.7 or higher
- PyTorch
- Torchvision
- Kaggle API (for dataset access)

### Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/jnlandu/group3-fs.git
   cd group3-fs/ResNet18-from-scratch-with-Mixup
   ```
2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Download the ImageNet dataset:**
    ```bash
    kaggle competitions download -c imagenet-object-localization-challenge
    unzip imagenet-object-localization-challenge.zip -d data
    ```


### Training the Model
To train the model, run the `main.py` script:
```bash
python main.py
```
This will start the training process, utilizing GPU acceleration if available. The training progress, including training and validation loss, will be displayed in the console.