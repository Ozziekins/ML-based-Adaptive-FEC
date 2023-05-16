import os
from sklearn.cluster import KMeans

import torch
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader

from backend.definitions import ROOT_DIR
from backend.models import NUM_CLUSTERS
from backend.models.Clustering import *
from backend.models.Clustering.ClusteringDataset import ClusteringDataset
from backend.models.Clustering.Model import Autoencoder

def train_ae(x, model_state):
    dataset = ClusteringDataset(x)

    # Dataloaders
    train_size = int(len(dataset) * TRAIN_SIZE)
    val_size = len(dataset) - train_size
    train, val = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train, batch_size=BATCH_SIZE)
    val_dataloader = DataLoader(val, batch_size=BATCH_SIZE)

    torch.set_float32_matmul_precision('medium')
    model = Autoencoder()
    model.load_state_dict(model_state)
    
    model.train()

    root_dir = os.path.join(ROOT_DIR, "store", "Autoencoder")
    trainer = pl.Trainer(max_epochs=MAX_EPOCHS, devices=1, default_root_dir=root_dir)
    trainer.fit(model, train_dataloader, val_dataloader) 
    torch.save(model.state_dict(), os.path.join(root_dir, 'model.pth'))    

    return model   

def cluster_train(model, data):    
    path = os.path.join(ROOT_DIR, "store", "cluster")    
    model = KMeans(n_clusters=NUM_CLUSTERS, init="k-means++", random_state=42, n_init=10)
    cluster_model = model.fit(data)
    os.makedirs(path, exist_ok=True)
    torch.save(cluster_model, os.path.join(path, 'model.pth'))
    return cluster_model