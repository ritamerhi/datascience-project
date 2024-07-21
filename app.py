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

#--------------------------------------------------------------------------------------------------

# from flask import Flask, render_template, request, jsonify
# import numpy as np
# import joblib

# app = Flask(__name__)

# # Load the machine learning model (adjust the path as necessary)
# model = joblib.load('model.pkl')

# @app.route('/')
# def home():
#     return render_template('index1.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.form
#         engine_size = float(data['engine_size'])
        
#         # Prepare the features for prediction
#         features = np.array([[engine_size]])  # Adjust according to your model's input format
        
#         # Make prediction
#         prediction = model.predict(features)
        
#         # Return the result
#         return jsonify(result=prediction[0])  # Adjust according to your model's output
#     except KeyError as e:
#         return jsonify(error=f"Missing data for {e.args[0]}"), 400
#     except Exception as e:
#         return jsonify(error=str(e)), 500

# if __name__ == '__main__':
#     app.run(debug=True)

#--------------------------------------------------------------------------------------------------

# from flask import Flask, request, jsonify, render_template
# import numpy as np
# import joblib

# app = Flask(__name__)

# # Load the pre-trained model
# model = joblib.load('model.pkl')

# @app.route('/')
# def home():
#     return render_template('index1.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Extract data from form
#         data = request.form.to_dict()
        
#         # Convert form data to features
#         features = np.array([[
#             float(data['engine_size']),
#             float(data['cylinders']),
#             float(data['fuel_consumption_city']),
#             float(data['fuel_consumption_hwy']),
#             float(data['fuel_consumption_comb']),
#             float(data['smog_level'])
#         ]])
        
#         # Make prediction
#         prediction = model.predict(features)
        
#         # Render result
#         return render_template('index.html', prediction_text='Predicted CO2 Emission: {:.2f} g/km'.format(prediction[0]))
    
#     except ValueError as e:
#         return render_template('index.html', prediction_text='Error: Invalid input. Please enter numeric values.')

# if __name__ == '__main__':
#     app.run(debug=True)


