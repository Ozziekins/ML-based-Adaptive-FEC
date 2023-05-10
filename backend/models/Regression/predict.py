
import torch
from backend.models import NUM_CLUSTERS


def regressor_predict(models, x, clusters):
    with torch.no_grad:
        def _predict(i, clusters):
            subset = [index for index, label in enumerate(clusters) if label == i]
            model = models[i]
            model.eval()
            return model(x[subset]).cpu().detach().numpy()
        res = [_predict(i, clusters) for i in range(NUM_CLUSTERS)]
        return res