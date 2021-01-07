import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
from matplotlib import pyplot as plt


# Select the GPUs to be used in this program
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# Define the model
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Training data
xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)


# Start training
history = model.fit(xs, ys, epochs=500)

# Predict one value

plt.plot(history.history['loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

print(model.predict([10.0]))
