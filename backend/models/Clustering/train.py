import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

from backend.models.Clustering import *
from backend.models.Clustering.ClusteringDataset import ClusteringDataset
from backend.models.Clustering.Model import Autoencoder


def train_ae(x):
    dataset = ClusteringDataset(x)

    # Dataloaders
    train_size = int(len(dataset) * TRAIN_SIZE)
    val_size = len(dataset) - TRAIN_SIZE
    train, val = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train, batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(val, batch_size=BATCH_SIZE)

    torch.set_float32_matmul_precision('medium')
    model = Autoencoder()
    trainer = pl.Trainer(max_epochs=MAX_EPOCHS, devices=1, default_root_dir=f'store/{type}')
    trainer.fit(model, train_dataloader, val_dataloader)
    return model.eval()