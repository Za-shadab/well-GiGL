from flask import Flask, request, jsonify
import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from flask_cors import CORS  # Import the CORS extension

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load models and scalers
gi_model = tf.keras.models.load_model('Glycemic Index Model.h5')
gl_model = tf.keras.models.load_model('Glycemic Load Model.h5')
scaler_gi = joblib.load('scaler.pkl1')
scaler_gl = joblib.load('scaler.pkl')

# Load LabelEncoders
label_encoder_gi = LabelEncoder()
label_encoder_gi.classes_ = np.array(['High', 'Low', 'Medium'])
label_encoder_gl = LabelEncoder()
label_encoder_gl.classes_ = np.array(['High', 'Low', 'Medium'])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data_gi = np.array([[
            data['calories'],
            data['proteins'],
            data['carbohydrates'],
            data['fats']
        ]])
        def classify_gl(gl):
            if gl <= 10:
                return 'Low'
            elif gl <= 19:
                return 'Medium'
            else:
                return 'High'

        # Predict Glycemic Index
        input_scaled_gi = scaler_gi.transform(input_data_gi)
        prediction_gi = gi_model.predict(input_scaled_gi)
        gi_value = round(float(prediction_gi[1][0][0]))
        category_index_gi = np.argmax(prediction_gi[0], axis=1)
        category_label_gi = label_encoder_gi.inverse_transform(category_index_gi)[0]
        confidence_score_gi = float(prediction_gi[0][0][category_index_gi][0] * 100)  
        
        # Predict Glycemic Load using predicted GI
        input_data_gl = np.array([[
            data['calories'],
            data['proteins'],
            data['carbohydrates'],
            data['fats'],
            gi_value  # Use predicted GI value
        ]])
        input_df_gl = pd.DataFrame(input_data_gl, columns=['calories', 'proteins (g)', 'carbohydrates (g)', 'fats (g)', 'glycemic_index'])
        input_scaled_gl = scaler_gl.transform(input_df_gl)
        prediction_gl = gl_model.predict(input_scaled_gl)
        gl_value = round(float(prediction_gl[1][0][0]), 1)
        gl_category_index = np.argmax(prediction_gl[0], axis=1)
        gl_category = label_encoder_gl.inverse_transform(gl_category_index)[0]
        
        return jsonify({
            'glycemic_index': {
                'value': gi_value,
                'category': category_label_gi,
                'confidence': f'{confidence_score_gi:.2f}%'
            },
            'glycemic_load': {
                'value': gl_value,
                'category': classify_gl(gl_value)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True,port= 5000)