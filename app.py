from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Model එක සහ Scaler එක Load කරගැනීම
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # පරිශීලකයා ලබාදෙන අගයන් ලබා ගැනීම
        weight = float(request.form['weight'])
        height = float(request.form['height'])
        age = float(request.form['age'])

        # BMI අගය ගණනය කිරීම
        bmi = weight / (height ** 2)

        # Features 4ම (Weight, Height, Age, BMI) Array එකකට සකස් කිරීම
        features = np.array([[weight, height, age, bmi]])
        
        # දත්ත Model එකට දීමට පෙර Scale කිරීම
        features_scaled = scaler.transform(features)
        
        # Prediction ලබාගැනීම
        prediction = model.predict(features_scaled)[0]
        
        # ප්‍රතිඵලය තීරණය කිරීම
        if prediction == 1:
            status = "High Health Risk"
        else:
            status = "Normal / Low Risk"

        return render_template('index.html', prediction_text=f"Status: {status} | BMI: {bmi:.2f}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)