import sys
import os
import numpy as np
import joblib

sys.path.append(os.path.abspath('..'))

# load model
model = joblib.load('app/artifacts/model.joblib')

def predict(feature1: float, feature2: float):

    features = np.array([[feature1, feature2]])
    prediction = model.predict(features)[0].astype(int)
    return prediction
