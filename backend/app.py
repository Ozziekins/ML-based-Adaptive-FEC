import json

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi import File, UploadFile
import os
import uvicorn

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


@app.post("/predict/json")
async def predict(request:Request):
    try:
        input = await request.json()
        dataset = pd.read_json(input)
        data = service.predict(np.expand_dims(dataset.to_numpy(),axis=0))
        return json.dumps(data.tolist())
    except Exception as ex:
        raise HTTPException(status_code=404, detail="There was an error getting predictions")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
