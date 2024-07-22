from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
import sklearn

# initializing flask app
app = Flask(__name__)

# loading the model into the app
model = joblib.load('model.pkl')
print(type(model))

# main page
@app.route('/')
def home():
    return render_template('index.html')

# page when 'predict' is clicked
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()  # retrieving form data submitted by the user
        
        # extracting features from form data and converting them to a numpy array
        features = np.array([[
            float(data['engine_size']),
            float(data['cylinders']),
            float(data['fuel_consumption_city']),
            float(data['fuel_consumption_hwy']),
            float(data['fuel_consumption_comb']),
            float(data['smog_level'])
        ]])
        
        # Making a prediction using the model we loaded
        prediction = model.predict(features)
        
        # Rendering the main page template with the prediction result
        return render_template('index.html', prediction_text='Predicted CO2 Emission: {:.2f} g/km'.format(prediction[0]))
    
    except Exception as e: # handling exceptions
        print(f"Error occurred: {e}") # getting info about the error to debug
        return render_template('index.html', prediction_text='An error occurred. Please try again.')


if __name__ == '__main__':
    app.run(debug=True)

