import tensorflow as tf
import os
from matplotlib import pyplot as plt

# Select the GPUs to be used in this program
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"


# Define callback class
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if logs.get('accuracy') > 0.99:
            print("\nReached 0.99 accuracy so cancelling training!")
            self.model.stop_training = True


# Declare callback object
callbacks = myCallback()
# Load  mnist dataset
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Normalize dataset
training_images = training_images / 255.0
test_images = test_images / 255.0

# Define model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compile and train model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])

plt.plot(history.history['loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

plt.plot(history.history['accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

print('end of file')
