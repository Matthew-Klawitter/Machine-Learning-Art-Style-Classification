from PIL import Image
from scipy import misc
from keras.models import Sequential
from keras.layers import Dense, Flatten
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
import cv2
import os


# An array containing the possible classifications
class_names = ['drawings', 'engraving', 'iconography', 'painting', 'sculpture']

# Collects image data and label pairs from a specified directory
def collect_data(img_dir):
    x = []
    y = []
    class_count = 0
    classes = os.listdir(img_dir)

    for image_class in classes:
        imgs = os.listdir(img_dir + "\\" + image_class)

        for image in imgs:
            img = cv2.imread(img_dir + "\\" + image_class + "\\" + image, 0)
            img = cv2.resize(img, (64,64))
            x.append(img)
            y.append(class_count)
        class_count += 1
    return [x,y]

# Raw training data parsed from images
training_data = collect_data("Images")
train_x = training_data[0]
train_y = training_data[1]

# Final numpy arrays for training
training_images = np.array(train_x) / 255.0
training_labels = np.array(train_y)

# Raw testing data parsed from images
testing_data = collect_data("Validation_Images")
test_x = testing_data[0]
test_y = testing_data[1]

# Final numpy arrays for testing
testing_images = np.array(test_x)
testing_labels = np.array(test_y)

# Prints shape of training data
print("Shape of training data:")
print(training_images.shape)
print(training_labels.shape)

# Creates a location to save training checkpoints along with the model
cp_path = "Trained_Model\\cp.ckpt"
cp_dir = os.path.dirname(cp_path)

# Establishes the checkpoint callback for saving the model in parts
cp_callback = tf.keras.callbacks.ModelCheckpoint(cp_path, save_weights_only=True, verbose=1)

# Creates a Sequential tensorflow neural network under the keras framework
model = Sequential()

# Adding layers to the model
model.add(Flatten(input_shape=(64,64)))
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dense(units=5, activation="softmax"))

# Compiles the model together, the last step to establishing the neural network
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Trains the values
model.fit(training_images, training_labels, epochs=30, batch_size=32, shuffle=True, callbacks=[cp_callback])

# Evaluates the testing samples
test_loss, test_acc = model.evaluate(testing_images, testing_labels, verbose=0)

print("Test accuracy:", test_acc)
print("Test loss:", test_loss)

predictions = model.predict(testing_images)
print(predictions[0])
print(class_names[np.argmax(predictions[0])])

print("Saving model to 'Trained_Model\\model.h5'")
model.save("Trained_Model\\model.h5")

print("Done!")