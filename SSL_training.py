
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

collate_fn = data.ImageCollateFunction(input_size=256, cj_prob=0.5, cj_bright=0.1,
                                       cj_hue=0.1, gaussian_blur=0.5,
                                       kernel_size=(25, 25), sigmas = (0.1, 2.0), 
                                       hf_prob=0.5, vf_prob=0.5, rr_prob=0.5,
                                       normalize={'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}



# build a PyTorch dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,                # pass the dataset to the dataloader
    batch_size=32,          # a large batch size helps with the learning
    shuffle=True,           # shuffling is important!
    collate_fn=collate_fn,  # apply transformations to the input images
    num_workers=32,         
)


import pytorch_lightning as pl
import torch.optim as optim

resnet = torchvision.models.resnet18()
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])

max_epochs = 1000
                                       
                                       
# Using SimCLR

                                       
class SimCLR(pl.LightningModule):
    def __init__(self, backbone, hidden_dim, out_dim):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, out_dim)
        self.criterion = NTXentLoss(temperature=0.5)

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


SimCLR_model = SimCLR(resnet, hidden_dim=512, out_dim=128)
trainer = pl.Trainer(max_epochs=max_epochs, auto_select_gpus=True, devices=1,
                     accelerator="gpu", accumulate_grad_batches=32)　
trainer.fit(
    SimCLR_model,
    dataloader
)

trainer.save_checkpoint("SimCLR_SSL_model.pt")


# Using TiCo


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

TiCo_model = TiCo(resnet, hidden_dim=512, out_dim=128)
trainer = pl.Trainer(max_epochs=max_epochs, auto_select_gpus=True, devices=1,
                     accelerator="gpu", accumulate_grad_batches=32)

trainer.fit(
    TiCo_model,
    dataloader
)

trainer.save_checkpoint("TiCo_SSL_model.pt")




