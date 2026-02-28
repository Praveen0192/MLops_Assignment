from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained and saved successfully!")