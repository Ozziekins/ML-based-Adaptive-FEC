import json

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi import File, UploadFile
import os

from backend.models.MLService import MLService

app = FastAPI()

service = MLService()


@app.post("/train")
def train(file: UploadFile = File(...)):    
    try:
        dataset = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=404, detail="There was an error getting training data")
    finally:
        file.file.close()
        
    service.train(dataset)


    return {"message": f"Models successfully trained!"}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        path = os.path.join("tmp", "predict.npy")
        with open(path, 'wb+') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error getting the rows for prediction"}

    try:
        dataset = np.fromfile(path)
        data = service.predict(dataset)
        return json.dumps(data.tolist())
    except Exception:
        return {"message": "Models should be trained before making prediction!"}

