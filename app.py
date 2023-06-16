import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import urllib.request
import tarfile
import os

# Download CIFAR-10 dataset
url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
filename = "cifar-10-python.tar.gz"
urllib.request.urlretrieve(url, filename)

# Extract CIFAR-10 dataset
with tarfile.open(filename, "r:gz") as tar:
    tar.extractall()
    
# Define dataset directory
dataset_dir = "cifar-10-batches-py"

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Preprocess the data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Define the CNN model architecture
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

# Compile the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=1)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print("Test Accuracy:", test_acc)

# Clean up downloaded dataset and extracted files
os.remove(filename)
os.system("rm -r " + dataset_dir)
