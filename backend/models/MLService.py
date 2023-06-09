import torch
import numpy as np

from backend.models import NUM_CLUSTERS
from backend.models.Clustering.load import load_autoencoder, load_cluster_model
from backend.models.Clustering.predict import autoencoder_embed
from backend.models.Clustering.train import cluster_train, train_ae
from backend.models.Preprocessing import Preprocessor
from backend.models.Regression.train import train_regressors
from backend.models.Regression.load import load_regressor
import time


class MLService:
    def __init__(self, ae=None, regressors=None, cluster_model=None):
        if regressors is None:
            regressors = {i: load_regressor(i) for i in range(NUM_CLUSTERS)}

        if ae is None:
            ae = load_autoencoder()

        if cluster_model is None:
            cluster_model = load_cluster_model()
        
        self.AE = ae
        self.regressors = regressors
        self.cluster_model = cluster_model
        self.preprocessor = Preprocessor()

    def train(self, data):
        # Preprocessing data
        x, y = self.preprocessor.preprocess(data)
        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
        
        # Preparing AutoEncoder
        start_time = time.time()
        ae = train_ae(x, self.AE.state_dict())
        ae_time = time.time()
        embeddings = ae(x).cpu().detach().numpy()

        # Preparing clusters          
        if(self.cluster_model is None):  
            cluster_model = cluster_train(self.cluster_model, embeddings)
        else: 
            cluster_model = self.cluster_model
        labels = cluster_model.predict(embeddings)
        cl_time = time.time()
        print(labels)
        #Preparing regressors
        regressors = train_regressors(x, y, labels, {k: v.state_dict() for k,v in self.regressors.items()})
        reg_time = time.time()
        # Rewriting the variables, so that we would not corrupt predict while training
        print("Auto Encoding Time : ", ae_time - start_time)
        print("Clustering Time : ", cl_time-ae_time)
        print("reg_time : ", reg_time-cl_time)
        self.AE, self.cluster_model, self.regressors = ae, cluster_model, regressors

    def predict(self, data):
        with torch.no_grad():
            if self.AE:
                x = self.preprocessor.preprocess_predict(data)
                start_time = time.time()
                embedding = autoencoder_embed(self.AE,x)
                ae_time = time.time()
                labels = self.cluster_model.predict(embedding)
                cl_time = time.time()        
                regressor = self.regressors[labels[0]]
                regressor.eval()                
                res = regressor(x) 
                reg_time = time.time()        
                print("Auto Encoding Time : ", ae_time - start_time)
                print("Clustering Time : ", cl_time-ae_time)
                print("reg_time : ", reg_time-cl_time)
                return res[0]
            else:
                raise Exception("Should be trained first!")
