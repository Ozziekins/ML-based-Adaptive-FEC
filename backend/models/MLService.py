import torch
import numpy as np

from backend.models import NUM_CLUSTERS
from backend.models.Clustering.load import load_autoencoder, load_kmeans
from backend.models.Clustering.predict import autoencoder_embed
from backend.models.Clustering.train import kmeans_train, train_ae
from backend.models.Preprocessing import Preprocessor
from backend.models.Regression.predict import regressor_predict
from backend.models.Regression.train import train_regressors
from backend.models.Regression.load import load_regressor


class MLService:
    def __init__(self, ae=None, regressors=None, kmeans=None):
        if regressors is None:
            regressors = {i: load_regressor(i) for i in range(NUM_CLUSTERS)}

        if ae is None:
            ae = load_autoencoder()

        if kmeans is None:
            kmeans = load_kmeans()
        
        self.AE = ae
        self.regressors = regressors
        self.kmeans = kmeans
        self.preprocessor = Preprocessor()

    def train(self, data):
        # Preprocessing data
        x, y = self.preprocessor.preprocess(data)
        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
        
        # Preparing AutoEncoder
        ae = train_ae(x, self.AE.state_dict())
        embeddings = ae(x).cpu().detach().numpy()

        # Preparing clusters    
        kmeans = kmeans_train(self.kmeans, embeddings)
        labels = kmeans.predict(embeddings)

        #Preparing regressors
        regressors = train_regressors(x, y, labels, {k: v.state_dict() for k,v in self.regressors.items()})
        
        # Rewriting the variables, so that we would not corrupt predict while training
        self.AE, self.kmeans, self.regressors = ae, kmeans, regressors

    def predict(self, data):
        with torch.no_grad():
            if self.AE:
                x = self.preprocessor.preprocess_predict(data)
                
                embedding = autoencoder_embed(self.AE,x)
                labels = self.kmeans.predict(embedding)

                res = regressor_predict(self.regressors,x,labels) 

                return np.vstack(res)
            else:
                raise Exception("Should be trained first!")
