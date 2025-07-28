# app.py

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from preprocessor import FraudPreprocessor

app = Flask(__name__)

# Load model with custom preprocessor class available
with open('best_fraud_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {
            'TRANSACTION_ID': [request.form['TRANSACTION_ID']],
            'TX_DATETIME': [request.form['TX_DATETIME']],
            'CUSTOMER_ID': [request.form['CUSTOMER_ID']],
            'TERMINAL_ID': [request.form['TERMINAL_ID']],
            'TX_AMOUNT': [float(request.form['TX_AMOUNT'])],
            'TX_TIME_SECONDS': [float(request.form['TX_TIME_SECONDS'])],
            'TX_TIME_DAYS': [float(request.form['TX_TIME_DAYS'])]
        }

        df_input = pd.DataFrame(input_data)
        prediction = model.predict(df_input)[0]
        result = "🚨 Fraudulent Transaction" if prediction == 1 else "✅ Genuine Transaction"
        return render_template('index.html', prediction_text=result)
    
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
