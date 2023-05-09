import torch
from sklearn.cluster import KMeans

from backend.models.Clustering.train import train_ae
from backend.models.Preprocessing import Preprocessor
from backend.models.Regression.train import train_regressors


class MLService:
    def __init__(self, ae=None, regressors=None, kmeans=None):
        if regressors is None:
            regressors = {}
        self.AE = ae
        self.regressors = regressors
        self.kmeans = kmeans

    def train(self, data):
        # Preprocessing data
        x, y = Preprocessor().preprocess(data)
        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
        # Preparing AutoEncoder
        ae = train_ae(x)
        embeddings = self.AE(x).cpu().detach().numpy()
        # Preparing clusters
        kmeans = KMeans(n_clusters=3, n_init='auto')
        labels = self.kmeans.fit_predict(embeddings)
        regressors = train_regressors(x, y, labels)
        # Rewriting the variables, so that we would not corrupt predict while training
        self.AE, self.kmeans, self.regressors = ae, kmeans, regressors

    def predict(self, x):
        embedding = self.AE(x).cpu().detach().numpy()
        label = self.kmeans.predoct(embedding)
        return self.regressors[label](x).cpu().detach().numpy()
