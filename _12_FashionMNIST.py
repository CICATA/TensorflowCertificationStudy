import tensorflow as tf
import os
from matplotlib import pyplot as plt


# Select the GPUs to be used in this program
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# Import fashion MNIST dataset
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Save example data as an image
plt.imshow(training_images[0])
plt.show()

# Normalize the images
training_images = training_images / 255.0
test_images = test_images / 255.0

# Define the model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

# Compile and train the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)

# Evaluate the model against test dataset
model.evaluate(test_images, test_labels)

# Predict values in test images dataset
classifications = model.predict(test_images)
print(classifications[0])

print('end of file')

