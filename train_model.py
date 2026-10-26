import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# පුංචි Sample Data ටිකක් (Weight, Height, Age -> Risk: 0=Normal, 1=High Risk)
data = {
    'weight': [50, 70, 95, 60, 110, 65, 85, 55],
    'height': [1.6, 1.75, 1.7, 1.65, 1.8, 1.6, 1.75, 1.55],
    'age':    [22, 25, 45, 30, 50,  28, 40,  24],
    'risk':   [0,  0,  1,  0,  1,   0,  1,   0]
}
df = pd.DataFrame(data)
X = df[['weight', 'height', 'age']]
y = df['risk']

# AI Model එක Train කිරීම සහ Save කිරීම
model = RandomForestClassifier(random_state=42)
model.fit(X, y)
joblib.dump(model, 'health_model.pkl')
print("Model Saved Successfully!")