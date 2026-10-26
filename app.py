from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# යාවත්කාලීන කළ Model එක Load කරගැනීම
model = joblib.load('model.pkl')

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

        # Backend එක තුළ BMI අගය ගණනය කිරීම
        bmi = weight / (height ** 2)

        # Features 4ම (Weight, Height, Age, BMI) මොඩලයට ලබාදීම
        features = np.array([[weight, height, age, bmi]])
        prediction = model.predict(features)[0]
        
        # ප්‍රතිඵලය තීරණය කිරීම (1 = High Risk, 0 = Low Risk)
        if prediction == 1:
            status = "High Health Risk"
            risk_percent = "75%"
        else:
            status = "Normal / Low Risk"
            risk_percent = "10%"

        return render_template('index.html', prediction_text=f"Status: {status} | Risk Level: {risk_percent} (BMI: {bmi:.2f})")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)