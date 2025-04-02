import numpy as np
import joblib
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import joblib

# Load model and scaler
model = tf.keras.models.load_model('Glycemic Index Model.h5')
scaler = joblib.load('scaler.pkl1')

# Define new input data
new_data = np.array([[116,	9.0,	20.0,	0.4]])  # Change values as needed
		
new_data_scaled = scaler.transform(new_data)

# Get predictions
new_prediction = model.predict(new_data_scaled)

# Extract GI value
gi_predicted = round(new_prediction[1][0][0])

# Extract GI category
gi_category_pred = new_prediction[0]

# Load Label Encoder classes
label_encoder_gi = LabelEncoder()
label_encoder_gi.classes_ = np.array(['High', 'Low', 'Medium'])
print("Encoded Categories Order:", label_encoder_gi.classes_)

# Convert category index to label
category_index = np.argmax(gi_category_pred, axis=1)
category_label = label_encoder_gi.inverse_transform(category_index)
confidence_score = gi_category_pred[0][category_index][0] * 100  

# Print results
print(f"Glycemic Index Prediction: {gi_predicted}")
print(f"Glycemic Index Category: {category_label[0]}")
print(f'Glycemic Index Category Confidence: {confidence_score:.2f}%')
print("Softmax Output:", gi_category_pred)
print("Predicted Index:", category_index)