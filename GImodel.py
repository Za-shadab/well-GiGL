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

correlation_matrix = data[['glycemic_index', 'glycemic_load', 'calories', 'proteins (g)', 'carbohydrates (g)', 'fats (g)']].corr()

plt.figure(figsize=(5, 5))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

def categorize_gi_new(gi):
    if gi <= 55:
        return 'Low'
    elif gi <= 69:
        return 'Medium'
    else:
        return 'High'

data['gi_category'] = data['glycemic_index'].apply(categorize_gi_new)

numeric_features = ['calories', 'proteins (g)', 'carbohydrates (g)', 'fats (g)']
X = data[numeric_features]

y_gi = data['glycemic_index']
y_gi_category = data['gi_category']

label_encoder_gi = LabelEncoder()
y_gi_category_encoded = label_encoder_gi.fit_transform(y_gi_category)
onehot_encoder = OneHotEncoder(sparse_output=False)
y_gi_category_onehot = onehot_encoder.fit_transform(y_gi_category_encoded.reshape(-1, 1))
y_gi_array = y_gi.to_numpy()
y_combined = np.column_stack((y_gi_category_onehot, y_gi_array))

X_train, X_test, y_train, y_test = train_test_split(X, y_combined, test_size=0.1, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'scaler.pkl1')


input_shape = X_train.shape[1]

inputs = tf.keras.Input(shape=(input_shape,))

x = tf.keras.layers.Dense(32, activation='relu')(inputs)
x = tf.keras.layers.Dense(32, activation='relu')(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)

num_categories = y_gi_category_onehot.shape[1]
output1 = tf.keras.layers.Dense(num_categories, activation='softmax', name='category_output')(x)
output2 = tf.keras.layers.Dense(1, activation='relu', name='gi_output')(x)
model = tf.keras.Model(inputs=inputs, outputs=[output1, output2])

y_train_category = y_train[:, :num_categories]
y_train_gi = y_train[:, num_categories]

y_test_category = y_test[:, :num_categories]
y_test_gi = y_test[:, num_categories]

def adjust_learning_rate():

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-6 * 10**(epoch / 20))

    optimizer = tf.keras.optimizers.Adam()

    model.compile(optimizer=optimizer,
              loss={'category_output': 'categorical_crossentropy', 'gi_output': 'huber_loss'},
              metrics={'category_output': 'accuracy', 'gi_output': 'mae'})

    history = model.fit(X_train_scaled,
                        {'category_output': y_train_category, 'gi_output': y_train_gi},
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
              loss={'category_output': 'categorical_crossentropy', 'gi_output': 'huber_loss'},
              metrics={'category_output': 'accuracy', 'gi_output': 'mae'})

model.fit(
    X_train_scaled,
    {'category_output': y_train_category, 'gi_output': y_train_gi},
    epochs=75,
    validation_data=(X_test_scaled, {'category_output': y_test_category, 'gi_output': y_test_gi}),
    verbose=2)

results = model.evaluate(X_test_scaled, {'category_output': y_test_category, 'gi_output': y_test_gi})

print(f"GI Category Accuracy: {results[3]:.4f}")
print(f"GI MAE: {results[4]:.4f}")

new_data = np.array([[105.0, 8.1, 18.5, 1.9]])
new_data_scaled = scaler.transform(new_data)

new_prediction = model.predict(new_data_scaled)
gi_new_prediction = round((new_prediction[1])[0][0])
gi_new_category = new_prediction[0]

category_index = np.argmax(gi_new_category, axis=1)
category_label = label_encoder_gi.inverse_transform(category_index)

print("Encoded Categories Order:", label_encoder_gi.classes_)
print('Glycemic Index Prediction: ', gi_new_prediction)
print('Glycemic Index Category:', category_label[0])
print('np bincount:',np.bincount(y_gi_category_encoded))
model.save('Glycemic Index Model.h5')