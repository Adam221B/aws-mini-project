import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# දත්ත සකස් කර ගැනීම
np.random.seed(42)
n_samples = 600
weight = np.random.normal(70, 15, n_samples)    # බර (kg)
height = np.random.normal(1.72, 0.1, n_samples)  # උස (meters)
age = np.random.randint(18, 65, n_samples)     # වයස

# BMI අගය ගණනය කිරීම
bmi = weight / (height ** 2)

# Risk තීරණය කිරීම
risk = []
for b, a in zip(bmi, age):
    if b > 27 or (b > 25 and a > 45):
        risk.append(1)  # High Risk
    else:
        risk.append(0)  # Low Risk

# DataFrame එකට Features 4 ම ඇතුළත් කිරීම
df = pd.DataFrame({'weight': weight, 'height': height, 'age': age, 'bmi': bmi, 'risk': risk})

# Features 4 ක් තෝරා ගැනීම (weight, height, age, bmi)
X = df[['weight', 'height', 'age', 'bmi']]
y = df['risk']

# දත්ත Scale කිරීම (Features 4 කට Scaler එක Fit වේ)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model එක Train කිරීම
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save කිරීම
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and Scaler updated with 4 features successfully!")