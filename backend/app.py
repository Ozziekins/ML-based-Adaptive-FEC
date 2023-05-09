from fastapi import FastAPI
from fastapi import File, UploadFile
import os

app = FastAPI()


@app.post("/train")
def train(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        path = os.path.join("tmp", file.filename)
        with open(path, 'wb+') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error getting training data"}
    finally:
        file.file.close()

    return {"message": f"Successfully uploaded {file.filename}"}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        path = os.path.join("tmp", "predict.csv")
        with open(path, 'wb+') as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error getting the rows for prediction"}
