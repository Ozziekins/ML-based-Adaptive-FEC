import os
import joblib
from sklearn.cluster import Birch

import torch

from backend.definitions import ROOT_DIR
from backend.models import NUM_CLUSTERS
from backend.models.Clustering.Model import Autoencoder


def load_autoencoder():
    PATH = os.path.join(ROOT_DIR, "store", "Autoencoder", 'model.pth')
    model = Autoencoder()
    if os.path.exists(PATH):    
        model.load_state_dict(torch.load(PATH))
    return model

def load_birch():
    PATH = os.path.join(ROOT_DIR, "store", "KMeans", 'model.pth')    
    model = None
    if os.path.exists(PATH):
        model = torch.load(PATH)
    if model is None:
        model = Birch(n_clusters=NUM_CLUSTERS)
    return model
