import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Load the trained model
model = load_model('Glycemic Load Model.h5')

# Load the saved scaler
scaler = joblib.load('scaler.pkl')

# Load the LabelEncoder (if needed)
label_encoder_gl = LabelEncoder()
label_encoder_gl.classes_ = np.array(['Low', 'Medium', 'High'])  # Ensure correct mapping

# Define new input data (example values)
new_data = np.array([[116,	9.0,	20.0,	0.4, 57]])
new_data_df = pd.DataFrame(new_data, columns=['calories', 'proteins (g)', 'carbohydrates (g)', 'fats (g)', 'glycemic_index'])
new_data_scaled = scaler.transform(new_data_df)
# Transform the new data using the saved scaler
new_data_scaled = scaler.transform(new_data)

# Make predictions
new_prediction = model.predict(new_data_scaled)

# Extract predictions
gl_new_prediction = round(new_prediction[1][0][0], 1)
gl_new_category = new_prediction[0]

# Convert category index to label
category_index = np.argmax(gl_new_category, axis=1)
category_label = label_encoder_gl.inverse_transform(category_index)

# Print results
print('Glycemic Load Prediction:', gl_new_prediction)
print('Glycemic Load Category:', category_label[0])
