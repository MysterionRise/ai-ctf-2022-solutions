import pickle
from tensorflow import keras

with open("model_ex.pkl", "rb") as f:
    model = pickle.load(f)
    print(model)
