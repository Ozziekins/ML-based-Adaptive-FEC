from Regression.Model import LossRatePredictor
from Regression.SequenceDataset import SequenceDataset
import torch
import pytorch_lightning as pl
import os
from torch.utils.data import random_split, DataLoader

file_path = 'store/checkpoint.ckpt'
def train_model(type, output_size, dataloader, val_dataloader=None, n_features=5):
    file_path = f'store/{type}/checkpoint.ckpt'
    torch.set_float32_matmul_precision('medium')
    hidden_dim = 8
    n_layers = 5
    if not os.path.exists(file_path):
        model = LossRatePredictor(n_features, hidden_dim, n_layers, output_size)
    else:
        model = LossRatePredictor.load_from_checkpoint(file_path)
    trainer = pl.Trainer(max_epochs=20, devices=1,default_root_dir=f'store/{type}')
    trainer.fit(model, dataloader, val_dataloader)

def load_model(type):
    file_path = f'store/{type}/checkpoint.ckpt'    
    if os.path.exists(file_path):
        model = LossRatePredictor.load_from_checkpoint(file_path)
        return model

def prepare_dataset(data, input_size, gap_size, output_size, train_size, batch_size=1024):
    dataset = SequenceDataset(data, input_size, gap_size, output_size)
    train_size = int(len(dataset)*train_size)
    val_size = len(dataset) - train_size
    train, val = random_split(dataset, [train_size, val_size])
    train_dataloader = DataLoader(train, batch_size=batch_size)
    val_dataloader = DataLoader(val, batch_size=batch_size)
    return train_dataloader, val_dataloader

def predict(input, model):
    model.eval()
    return model(input)
