import os

import torch

from backend.definitions import ROOT_DIR
from backend.models.Clustering.Model import Autoencoder


def load_autoencoder():
    PATH = os.path.join(ROOT_DIR, "store", "Autoencoder", 'model.pth')
    model = Autoencoder()
    if os.path.exists(PATH):
        model.load_state_dict(torch.load(PATH))
    return model


def load_cluster_model():
    PATH = os.path.join(ROOT_DIR, "store", "cluster", 'model.pth')
    model = None
    if os.path.exists(PATH):
        model = torch.load(PATH)    
    return model
