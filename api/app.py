import joblib
import numpy as np
import torch
from fastapi import FastAPI, Request
from sklearn.preprocessing import StandardScaler

from ..model.model import BreastCancerNN

MODEL_PATH = "D:/Development/MLPractice/Kaggle/learn-breast-cancer-classification/model/BreastCancerNN.pth"
SCALER_PATH = "D:/Development/MLPractice/Kaggle/learn-breast-cancer-classification/model/BreastCancer_Scaler.bin"

# Create the network instance
breast_cancer_model = BreastCancerNN(30, 1)

# Load the weights
weights = torch.load(MODEL_PATH)

# Populate the weights on the network
breast_cancer_model.load_state_dict(weights)

# Set the network in evaluation mode
breast_cancer_model.eval()

# Load the scalar which was saved during training
scaler : StandardScaler = joblib.load(SCALER_PATH)

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/predict")
async def predict(request: Request):
    
    # Example input to the API. This is similar to the records in sklearn.datasets.load_breast_cancer() but without the target.
    # X = {
    #     "param1": 1.799e+01,
    #     "param2": 1.038e+01,
    #     "param3": 1.228e+02,
    #     "param4": 1.001e+03,
    #     "param5": 1.184e-01,
    #     "param6": 2.776e-01,
    #     "param7": 3.001e-01,
    #     "param8": 1.471e-01,
    #     "param9": 2.419e-01,
    #     "param0": 7.871e-02,
    #     "param10": 1.095e+00,
    #     "param11": 9.053e-01,
    #     "param12": 8.589e+00,
    #     "param13": 1.534e+02,
    #     "param14": 6.399e-03,
    #     "param15": 4.904e-02,
    #     "param16": 5.373e-02,
    #     "param17": 1.587e-02,
    #     "param18": 3.003e-02,
    #     "param19": 6.193e-03,
    #     "param20": 2.538e+01,
    #     "param21": 1.733e+01,
    #     "param22": 1.846e+02,
    #     "param23": 2.019e+03,
    #     "param24": 1.622e-01,
    #     "param25": 6.656e-01,
    #     "param26": 7.119e-01,
    #     "param27": 2.654e-01,
    #     "param28": 4.601e-01,
    #     "param29": 1.189e-01
    # }
    
    # Get the input from the request body
    X = await request.json()
    
    # Convert it to numpy array
    array = np.zeros(shape=(1, len(X)), dtype=float)
    array[0] = np.asarray(list(X.values()), dtype=float)
    
    # Transform using the scaler which was saved while training
    array = torch.FloatTensor(scaler.transform(array))
    
    prediction = None
    
    # Predict and return the output
    with torch.inference_mode():
        y_prediction = breast_cancer_model(array)
        
        # Map to the class
        prediction = "B" if torch.round(torch.sigmoid(y_prediction)).squeeze().item() == 0 else "M"
    
    return prediction