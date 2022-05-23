import pickle
import pandas as pd

with open("model_ex.pkl", "rb") as f:
    source_model = pickle.load(f)
    print(source_model.summary())

inner = source_model.layers[0]
print(inner.summary())

df = pd.read_csv("objects.csv")
print(df.head)

for _, row in df.iterrows():
    x = row.iloc[1:].to_numpy()
    x = x.reshape(1, -1)
    print(x)
    print(inner.predict(x))