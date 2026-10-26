import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. BMI සහ ආන්තික අගයන් (Extreme values) ඇතුළත් කළ Dataset එක
data = {
    'weight': [39, 85, 60, 45, 95, 70, 38, 110, 55, 90],
    'height': [1.60, 1.75, 1.70, 1.55, 1.80, 1.65, 1.60, 1.75, 1.68, 1.82],
    'age': [25, 45, 30, 22, 50, 35, 26, 55, 29, 48],
    'risk_level': [1, 1, 0, 0, 1, 0, 1, 1, 0, 1]  # 39kg (Underweight) සඳහා 1 (High Risk) ලෙස ලබා දී ඇත
}

df = pd.DataFrame(data)

# 2. BMI අගය ගණනය කර Column එකක් ලෙස එකතු කිරීම
df['bmi'] = df['weight'] / (df['height'] ** 2)

# 3. Features (X) සහ Target (y) වෙන් කරගැනීම (BMI එකත් සමඟ)
X = df[['weight', 'height', 'age', 'bmi']]
y = df['risk_level']

# 4. Random Forest Model එක Train කිරීම
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 5. පුහුණු කළ මොඩලය Save කරගැනීම
joblib.dump(model, 'model.pkl')
print("Updated BMI-based Model trained and saved successfully as 'model.pkl'!")