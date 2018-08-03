from PIL import Image
from scipy import misc
from keras.models import Sequential
from keras.layers import Dense, Flatten
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import cv2
import os


# Directory that holds image data
class_dir = os.listdir("Images")
test_dir = os.listdir("Validation_Images")

# An array containing the possible classifications
class_names = ['drawings', 'engraving', 'iconography', 'painting', 'sculpture']

train_x = [] # An array that holds image data
train_y = [] # An array that holds the label/class a image belongs to
test_x = [] # An array that holds test image data
test_y = [] # An array that holds the label/class a test image belongs to

# Collects image data and label pairs from the training sample
class_count = 0
for image_class in class_dir:
    imgs = os.listdir("Images\\" + image_class)

    for image in imgs:
        img = cv2.imread("Images\\" + image_class + "\\" + image, 0)
        img = cv2.resize(img, (64,64))
        train_x.append(img)
        train_y.append(class_count)
    class_count += 1

# Collects image data and label pairs from the testing/validation sample
class_count = 0
for image_class in test_dir:
    imgs = os.listdir("Validation_Images\\" + image_class)

    for image in imgs:
        img = cv2.imread("Validation_Images\\" + image_class + "\\" + image, 0)
        img = cv2.resize(img, (64,64))
        test_x.append(img)
        test_y.append(class_count)
    class_count += 1

# The final organized, numpy datasets
training_images = np.array(train_x) / 255.0
training_labels = np.array(train_y)

testing_images = np.array(test_x)
testing_labels = np.array(test_y)

print(training_images.shape)
print(training_labels.shape)


# Creates a Sequential tensorflow neural network under the keras framework
model = Sequential()

# Adding layers to the model
model.add(Flatten(input_shape=(64,64)))
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dense(units=5, activation="softmax"))

# Compiles the model together, the last step to establishing the neural network
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Trains the values
model.fit(training_images, training_labels, epochs=20, batch_size=32, shuffle=True)

# Evaluates the testing samples
test_loss, test_acc = model.evaluate(testing_images, testing_labels, verbose=0)

print("Test accuracy:", test_acc)

predictions = model.predict(testing_images)
print(predictions[0])
print(np.argmax(predictions[0]))


# Serializes the model to json format
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)

print("Done!")