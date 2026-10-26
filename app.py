from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# AI Model එක Memory එකට Load වීම
model = joblib.load('health_model.pkl')

# 1. බ්‍රව්සරයෙන් එන අයට HTML Dashboard එක පෙන්වන Route එක
@app.route('/')
def home():
    return render_template('index.html')

# 2. AI Model එකෙන් Prediction එක ලබා දෙන API Route එක
@app.route('/predict-risk', methods=['POST'])
def predict_risk():
    try:
        data = request.get_json()
        weight = float(data['weight'])
        height = float(data['height'])
        age = float(data['age'])
        
        input_data = np.array([[weight, height, age]])
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] * 100
        
        result_text = "High Health Risk" if prediction == 1 else "Normal / Low Risk"
        
        return jsonify({
            "status": "success",
            "prediction": result_text,
            "risk_percentage": round(probability, 2),
            "server": "AWS Cloud Ubuntu EC2"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)