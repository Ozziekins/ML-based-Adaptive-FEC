
import torch
from backend.models import NUM_CLUSTERS


def autoencoder_embed(model, x):
    with torch.no_grad:
        model.eval()
        embedding = model(x).cpu().detach().numpy()
        return embedding