from backend.models import PREDICT_SIZE
from backend.models.Regression import *
from backend.models.Regression.Model import LossRatePredictor
from backend.models.Regression.SequenceDataset import SequenceDataset
import torch
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import random_split, DataLoader

file_path = 'store/checkpoint.ckpt'


def train_regressors(x, y, labels):
    models = {}
    for i in range(3):
        subset = np.argwhere(labels == i)
        dataset = SequenceDataset(x[subset], y[subset])
        models[i] = train_regressor(dataset)
    return models


def train_regressor(dataset):
    # Dataloaders
    train_size = int(len(dataset) * TRAIN_SIZE)
    val_size = len(dataset) - TRAIN_SIZE
    train, val = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train, batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(val, batch_size=BATCH_SIZE)

    torch.set_float32_matmul_precision('medium')
    model = LossRatePredictor(N_FEATURES, HIDDEN_DIM, N_LAYERS, PREDICT_SIZE)
    trainer = pl.Trainer(max_epochs=20, devices=1)
    trainer.fit(model, train_dataloader, val_dataloader)
    return model
