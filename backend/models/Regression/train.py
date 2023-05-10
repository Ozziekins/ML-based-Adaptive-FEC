import os.path

from backend.models import PREDICT_SIZE, NUM_CLUSTERS
from backend.models.Regression import *
from backend.models.Regression.Model import LossRatePredictor
from backend.models.Regression.SequenceDataset import SequenceDataset
from backend.definitions import ROOT_DIR
import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader


def train_regressors(x, y, labels, state_dicts):
    models = {}
    for i in range(NUM_CLUSTERS):
        subset = [index for index, label in enumerate(labels) if label == i]
        dataset = SequenceDataset(x[subset], y[subset])
        models[i] = train_regressor(dataset, i, state_dicts[i])
    return models
    
def train_regressor(dataset, index, state_dict):
    # Dataloaders
    train_size = int(len(dataset) * TRAIN_SIZE)
    val_size = len(dataset) - train_size
    train, val = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train, batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(val, batch_size=BATCH_SIZE)

    torch.set_float32_matmul_precision('medium')

    model = LossRatePredictor(N_FEATURES, HIDDEN_DIM, N_LAYERS, PREDICT_SIZE)
    model.load_state_dict(state_dict)    
    model.train()

    root_dir = os.path.join(ROOT_DIR, "store", "Regressor", str(index))
    trainer = pl.Trainer(max_epochs=20, devices=1, default_root_dir=root_dir)
    trainer.fit(model, train_dataloader, val_dataloader)
    torch.save(model.state_dict(), os.path.join(root_dir, 'model.pth'))
    return model
