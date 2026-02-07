import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Sample dataset (fake demo data)
data = {
    "rainfall": [200, 300, 150, 400, 250, 350],
    "temperature": [30, 28, 35, 26, 32, 27],
    "fertilizer": [40, 60, 20, 80, 50, 70],
    "yield": [2.5, 3.5, 1.8, 4.2, 3.0, 3.8]
}

df = pd.DataFrame(data)

X = df[["rainfall", "temperature", "fertilizer"]]
y = df["yield"]

model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "crop_model.pkl")

print("Model trained and saved!")
