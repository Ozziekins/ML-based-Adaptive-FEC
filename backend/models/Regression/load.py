import os

import torch

from backend.definitions import ROOT_DIR
from backend.models import PREDICT_SIZE
from backend.models.Clustering.Model import Autoencoder
from backend.models.Regression import HIDDEN_DIM, N_FEATURES, N_LAYERS
from backend.models.Regression.Model import LossRatePredictor

def load_regressor(index):
    PATH = os.path.join(ROOT_DIR, "store", "Regressor", str(index), 'model.pth')
    model = LossRatePredictor(N_FEATURES, HIDDEN_DIM, N_LAYERS, PREDICT_SIZE)
    if os.path.exists(PATH):    
        model.load_state_dict(torch.load(PATH))
    return model
