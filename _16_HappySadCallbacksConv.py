import tensorflow as tf
import os
from matplotlib import pyplot as plt
import wget
import zipfile
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Select the GPUs to be used in this program
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

# Load data
data_dir = 'happy-or-sad'

if not os.path.exists(data_dir + '.zip'):
    # Download dataset
    url = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip'
    filename = wget.download(url)
    zip_ref = zipfile.ZipFile(filename, 'r')
    zip_ref.extractall(data_dir)
    zip_ref.close()

# Set callback to stop trainning when desired accuracy is achieved
DESIRED_ACCURACY = 0.999


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if logs.get('accuracy') > DESIRED_ACCURACY:
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True


callbacks = myCallback()

# Define model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

# Generate training dataset from directory
train_datagen = ImageDataGenerator(rescale=1 / 255)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=10,
    class_mode='binary')

# Expected output: 'Found 80 images belonging to 2 classes'

# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    verbose=1,
    callbacks=[callbacks])

# Plot results
plt.plot(history.history['accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

plt.plot(history.history['loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

print('end of file')
