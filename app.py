from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
import sklearn


app = Flask(__name__)

model = joblib.load('model.pkl')
print(type(model))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        
        features = np.array([[
            float(data['engine_size']),
            float(data['cylinders']),
            float(data['fuel_consumption_city']),
            float(data['fuel_consumption_hwy']),
            float(data['fuel_consumption_comb']),
            float(data['smog_level'])
        ]])
        
        prediction = model.predict(features)
        
        return render_template('index.html', prediction_text='Predicted CO2 Emission: {:.2f} g/km'.format(prediction[0]))
    
    except Exception as e:
        print(f"Error occurred: {e}")
        return render_template('index.html', prediction_text='An error occurred. Please try again.')


if __name__ == '__main__':
    app.run(debug=True)

