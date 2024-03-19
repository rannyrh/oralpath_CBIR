
import pandas as pd
import numpy as np
from PIL import Image
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
import flash
from flash.core.classification import LabelsOutput
from flash.core.data.utils import download_data
from flash.image import ImageClassificationData, ImageClassifier
import torchvision
from torchvision import transforms
from lightly.loss.tico_loss import TiCoLoss
from lightly.models.modules.heads import TiCoProjectionHead
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead


import lightly.data as data

# create a dataset from your image folder
dataset = data.LightlyDataset(input_dir='./finetune/lightly/')

# Define collate function with image augmentation parameters
collate_fn = data.ImageCollateFunction(input_size=256, cj_prob=0.5, cj_bright=0.1,
                                       cj_hue=0.1, gaussian_blur=0.5,
                                       kernel_size=(25, 25), sigmas = (0.1, 2.0), 
                                       hf_prob=0.5, vf_prob=0.5, rr_prob=0.5,
                                       normalize={'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

#input_size=256: Specifies the size of each input images in pixels.
#cj_prob=0.5: Probability of applying color jittering during data augmentation.
#cj_bright=0.1: Strength of brightness adjustment during color jittering.
#cj_hue=0.1: Strength of hue adjustment during color jittering.
#gaussian_blur=0.5: Probability of applying Gaussian blur during data augmentation.
#kernel_size=(25, 25): Size of the Gaussian blur kernel.
#sigmas=(0.1, 2.0): Range of standard deviations for Gaussian blur.
#hf_prob=0.5: Probability of applying horizontal flip during data augmentation.
#vf_prob=0.5: Probability of applying vertical flip during data augmentation.
#rr_prob=0.5: Probability of applying random rotation during data augmentation.
#normalize={'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}: Mean and standard deviation used for normalization of images.
#other default hyperparameters of lightly.data.ImageCollateFunction are described in https://docs.lightly.ai/self-supervised-learning/lightly.data.html


# Build a PyTorch dataloader with specified parameters
dataloader = torch.utils.data.DataLoader(
    dataset,                # pass the dataset to the dataloader
    batch_size=32,          # a large batch size helps with the learning
    shuffle=True,           # shuffling is important!
    collate_fn=collate_fn,  # apply transformations to the input images
    num_workers=32,         
)

#batch_size=32: Number of samples per batch during training.
#shuffle=True: Randomly shuffles the dataset before each epoch.
#num_workers=32: Number of subprocesses to use for data loading.

# Import required libraries
import pytorch_lightning as pl
import torch.optim as optim

# Initialize ResNet model and remove the last layer
resnet = torchvision.models.resnet18()
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])

max_epochs = 1000
#max_epochs = 1000: Maximum number of epochs to train the models.                                       
                                       
# Using SimCLR

# Define SimCLR LightningModule class
class SimCLR(pl.LightningModule):
    def __init__(self, backbone, hidden_dim, out_dim):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, out_dim)
        self.criterion = NTXentLoss(temperature=0.5)
        
        #hidden_dim=512: Dimensionality of the hidden layers in the projection head.
        #out_dim=128: Dimensionality of the output space of the projection head.
        #temperature=0.5: Temperature parameter for NT-Xent loss
  
    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        return loss

    def configure_optimizers(self):
        optimizer = flash.core.optimizers.LARS(self.parameters(), lr=1.2) 
        return optimizer

        #lr=1.2: Learning rate for the optimizer. LARS stands for Layer-wise Adaptive Rate Scaling.

# Initialize SimCLR model
SimCLR_model = SimCLR(resnet, hidden_dim=512, out_dim=128)

# Initialize trainer for SimCLR
trainer = pl.Trainer(max_epochs=max_epochs, auto_select_gpus=True, devices=1,
                     accelerator="gpu", accumulate_grad_batches=32)　

#auto_select_gpus=True: Automatically selects available GPUs.
#devices=1: Number of GPU devices to use.
#accelerator="gpu": Specifies using the GPU for acceleration.
#accumulate_grad_batches=32: Number of batches to accumulate gradients over before performing a backward pass.

# Train SimCLR model
trainer.fit(
    SimCLR_model,
    dataloader
)

# Save trained SimCLR model checkpoint
trainer.save_checkpoint("SimCLR_SSL_model.pt")

#"SimCLR_SSL_model.pt": Filename to save the trained model checkpoint.


# Using TiCo

# Define TiCo LightningModule class
class TiCo(pl.LightningModule):
    def __init__(self, backbone, hidden_dim, out_dim):
        super().__init__()
        self.backbone = backbone
        self.projection_head = TiCoProjectionHead(hidden_dim, hidden_dim, out_dim)
        self.criterion = TiCoLoss()

    def forward(self, x):
        h = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(h)
        return z

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        return loss

    def configure_optimizers(self):
        optimizer = flash.core.optimizers.LARS(self.parameters(), lr=1.2) 
        return optimizer

# Initialize TiCo model
TiCo_model = TiCo(resnet, hidden_dim=512, out_dim=128)

# Initialize trainer for TiCo
trainer = pl.Trainer(max_epochs=max_epochs, auto_select_gpus=True, devices=1,
                     accelerator="gpu", accumulate_grad_batches=32)

# Train TiCo model
trainer.fit(
    TiCo_model,
    dataloader
)

# Save trained TiCo model checkpoint
trainer.save_checkpoint("TiCo_SSL_model.pt")




