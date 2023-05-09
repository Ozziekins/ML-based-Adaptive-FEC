import os.path

from backend.models import PREDICT_SIZE, NUM_CLUSTERS
from backend.models.Regression import *
from backend.models.Regression.Model import LossRatePredictor
from backend.models.Regression.SequenceDataset import SequenceDataset
from backend.definitions import ROOT_DIR
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

file_path = 'store/checkpoint.ckpt'


def train_regressors(x, y, labels):
    models = {}
    for i in range(NUM_CLUSTERS):
        subset = [index for index, label in enumerate(labels) if label == i]
        dataset = SequenceDataset(x[subset], y[subset])
        models[i] = train_regressor(dataset)
    return models


def train_regressor(dataset):
    # Dataloaders
    train_size = int(len(dataset) * TRAIN_SIZE)
    val_size = len(dataset) - train_size
    train, val = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train, batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(val, batch_size=BATCH_SIZE)

    torch.set_float32_matmul_precision('medium')
    model = LossRatePredictor(N_FEATURES, HIDDEN_DIM, N_LAYERS, PREDICT_SIZE)

    root_dir = os.path.join(ROOT_DIR, "store", "Regressor")
    trainer = pl.Trainer(max_epochs=20, devices=1, default_root_dir=root_dir)
    trainer.fit(model, train_dataloader, val_dataloader)
    return model
