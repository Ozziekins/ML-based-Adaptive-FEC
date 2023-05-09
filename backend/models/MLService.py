import torch
from sklearn.cluster import KMeans
import numpy as np

from backend.models import NUM_CLUSTERS
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
        self.preprocessor = Preprocessor()

    def train(self, data):
        # Preprocessing data
        x, y = self.preprocessor.preprocess(data)
        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
        # Preparing AutoEncoder
        ae = train_ae(x)
        embeddings = ae(x).cpu().detach().numpy()
        # Preparing clusters
        kmeans = KMeans(n_clusters=NUM_CLUSTERS, n_init='auto')
        labels = kmeans.fit_predict(embeddings)
        regressors = train_regressors(x, y, labels)
        # Rewriting the variables, so that we would not corrupt predict while training
        self.AE, self.kmeans, self.regressors = ae, kmeans, regressors

    def predict(self, data):
        if self.AE:
            x = self.preprocessor.preprocess_predict(data)

            def _predict(i, clusters):
                subset = [index for index, label in enumerate(clusters) if label == i]
                return self.regressors[i](x[subset]).cpu().detach().numpy()

            embedding = self.AE(x).cpu().detach().numpy()
            labels = self.kmeans.predict(embedding)

            res = [_predict(i, labels) for i in range(NUM_CLUSTERS)]

            return np.vstack(res)
        else:
            raise Exception("Should be trained first!")
