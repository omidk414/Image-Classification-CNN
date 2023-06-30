#!/usr/bin/env python
# coding: utf-8

# In[7]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import urllib.request
import tarfile
import os


# In[8]:


# Download CIFAR-10 dataset
url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
filename = "cifar-10-python.tar.gz"
urllib.request.urlretrieve(url, filename)


# In[9]:


# Extract CIFAR-10 dataset
with tarfile.open(filename, "r:gz") as tar:
    tar.extractall()


# In[10]:


# Define dataset directory
dataset_dir = "cifar-10-batches-py"


# In[11]:


# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()


# In[12]:


# Preprocess the data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


# In[13]:


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


# In[14]:


# Compile the model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)


# In[ ]:


# Train the model
model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=1)


# In[ ]:


# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
print("Test Accuracy:", test_acc)


# In[ ]:


# Clean up downloaded dataset and extracted files
os.remove(filename)
os.system("rm -r " + dataset_dir)

