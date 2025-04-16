import joblib

# load model
model = joblib.load('app/artifacts/model.joblib')

# predict function
def predict(feature1: float, feature2: float) -> int:
    input_features = [[feature1, feature2]]
    prediction = model.predict(input_features)[0]
    return int(prediction)
