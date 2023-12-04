# Self-Supervised Learning for CBIR in oral pathology

This repository contains code for training self-supervised learning models for oral histopathological images using PyTorch Lightning. 
The code includes implementations of the SimCLR and TiCo models with ResNet18 as the backbone.

## Overview

The code provided trains two types of self-supervised learning models: SimCLR and TiCo. These models are designed to learn representations from unlabeled images in an unsupervised manner. The code leverages PyTorch Lightning, a lightweight PyTorch wrapper, for easier training and reproducibility.

## Requirements

- Python 3.6 or later
- PyTorch
- PyTorch Lightning
- torchvision
- flash (from PyTorch Lightning)
- lightly

## Installation

1. Clone this repository.
2. Install the required packages using `pip`:
   ```bash
   pip install torch torchvision pytorch-lightning flash lightly

## Data Preparation
Place your unlabeled image dataset in the ./finetune/lightly/ directory. The images will be automatically processed and augmented during training.

## Training the Models

Set the `max_epochs` variable to define the number of training epochs.

Run the training script:
```
python SSL_training.py
```
The trained models are saved as `SimCLR_SSL_model.pt` and `TiCo_SSL_model.pt` for SimCLR and TiCo, respectively.

