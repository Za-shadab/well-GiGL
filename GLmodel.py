import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import joblib

from keras.callbacks import EarlyStopping
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv('nutrition food dataset - modified.csv')
print(data)

def categorize_gl_new(gl):
    if gl <= 10:
        return 'Low'
    elif gl <= 19:
        return 'Medium'
    else:
        return 'High'

data['gl_category'] = data['glycemic_load'].apply(categorize_gl_new)

numeric_features = ['calories', 'proteins (g)', 'carbohydrates (g)', 'fats (g)', 'glycemic_index']
X = data[numeric_features]

y_gl = data['glycemic_load']
y_gl_category = data['gl_category']

label_encoder_gl = LabelEncoder()
y_gl_category_encoded = label_encoder_gl.fit_transform(y_gl_category)
onehot_encoder = OneHotEncoder(sparse_output=False)
y_gl_category_onehot = onehot_encoder.fit_transform(y_gl_category_encoded.reshape(-1, 1))
y_gl_array = y_gl.to_numpy()
y_combined = np.column_stack((y_gl_category_onehot, y_gl_array))

X_train, X_test, y_train, y_test = train_test_split(X, y_combined, test_size=0.1, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl')

input_shape = X_train.shape[1]

inputs = tf.keras.Input(shape=(input_shape,))

x = tf.keras.layers.Dense(32, activation='relu')(inputs)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)

num_categories = y_gl_category_onehot.shape[1]
output1 = tf.keras.layers.Dense(num_categories, activation='softmax', name='category_output')(x)
output2 = tf.keras.layers.Dense(1, activation='relu', name='gl_output')(x)
model = tf.keras.Model(inputs=inputs, outputs=[output1, output2])

y_train_category = y_train[:, :num_categories]
y_train_gl = y_train[:, num_categories]

y_test_category = y_test[:, :num_categories]
y_test_gl = y_test[:, num_categories]
def adjust_learning_rate():

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-6 * 10**(epoch / 20))

    optimizer = tf.keras.optimizers.Adam()

    model.compile(optimizer=optimizer,
              loss={'category_output': 'categorical_crossentropy', 'gl_output': 'huber_loss'},
              metrics={'category_output': 'accuracy', 'gl_output': 'mae'})

    history = model.fit(X_train_scaled,
                        {'category_output': y_train_category, 'gl_output': y_train_gl},
                        epochs=100,
                        callbacks=[lr_schedule],
                        verbose=0)

    return history
lr_history = adjust_learning_rate()
plt.semilogx(lr_history.history['lr'], lr_history.history['loss'])
plt.axis([1e-6, 1e-1, 0, 45])
plt.show()

model = tf.keras.Model(inputs=inputs, outputs=[output1, output2])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
              loss={'category_output': 'categorical_crossentropy', 'gl_output': 'huber_loss'},
              metrics={'category_output': 'accuracy', 'gl_output': 'mae'})

model.fit(
    X_train_scaled,
    {'category_output': y_train_category, 'gl_output': y_train_gl},
    epochs=75,
    validation_data=(X_test_scaled, {'category_output': y_test_category, 'gl_output': y_test_gl}),
    verbose=2)

results = model.evaluate(X_test_scaled, {'category_output': y_test_category, 'gl_output': y_test_gl})

print(f"GL Category Accuracy: {results[3]:.4f}")
print(f"GL MAE: {results[4]:.4f}")

new_data = np.array([[391.0,	10.7,	27.0,	26.5, 35.0]])
new_data_scaled = scaler.transform(new_data)

new_prediction = model.predict(new_data_scaled)
gl_new_prediction = round((new_prediction[1])[0][0], 1)
gl_new_category = new_prediction[0]

category_index = np.argmax(gl_new_category, axis=1)
category_label = label_encoder_gl.inverse_transform(category_index)

print("Encoded Categories Order:", label_encoder_gl.classes_)
print('Glycemic Load Prediction: ', gl_new_prediction)
print('Glycemic Load Category:', category_label[0])
model.save('Glycemic Load Model.h5') 