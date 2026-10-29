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

# 1. වඩාත් නිවැරදි සහ විශාල Dataset එකක් සාදා ගැනීම (Realistic Data)
np.random.seed(42)
n_samples = 500

height = np.random.normal(165, 10, n_samples)  # Mean 165cm
weight = np.random.normal(70, 15, n_samples)   # Mean 70kg
age = np.random.randint(18, 70, n_samples)

# BMI එක මත පදනම්ව Realistic Risk Level එකක් සැකසීම (High BMI + High Age = High Risk)
bmi = weight / ((height / 100) ** 2)
risk = []
for b, a in zip(bmi, age):
    if b > 28 or (b > 25 and a > 50):
        risk.append(1)  # High Risk
    else:
        risk.append(0)  # Low Risk

df = pd.DataFrame({'height': height, 'weight': weight, 'age': age, 'risk': risk})

# 2. Features සහ Target වෙන් කිරීම
X = df[['height', 'weight', 'age']]
y = df['risk']

# 3. Data Scale කිරීම (Scaling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Model එක Train කිරීම
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Model එක සහ Scaler එක Save කිරීම
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')  # Scaler එකද Save කරගත යුතුය

print("Model and Scaler updated successfully!")