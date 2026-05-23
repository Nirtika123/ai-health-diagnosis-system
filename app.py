import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = pd.read_csv("dataset/dataset.csv")

# Features and target
X = data.drop("disease", axis=1)
y = data["disease"]

# Train model
model = DecisionTreeClassifier()
model.fit(X, y)

print("=== AI Health Diagnosis System ===")

# User input
fever = int(input("Fever (1/0): "))
cough = int(input("Cough (1/0): "))
headache = int(input("Headache (1/0): "))
nausea = int(input("Nausea (1/0): "))

# Prediction
symptoms = [[fever, cough, headache, nausea]]
prediction = model.predict(symptoms)

print("Predicted Disease:", prediction[0])
