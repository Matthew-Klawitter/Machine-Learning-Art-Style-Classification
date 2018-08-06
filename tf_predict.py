from keras.models import load_model
from sklearn.metrics import confusion_matrix
import argparse
import cv2
import numpy as np
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
            img = cv2.imread(img_dir + "\\" + image_class + "\\" + image)
            img = cv2.resize(img, (64,64))
            x.append(img)
            y.append(class_count)
        class_count += 1
    return [x,y]

# Raw testing data parsed from images
testing_data = collect_data("Validation_Images")
test_x = testing_data[0]
test_y = testing_data[1]

# Final numpy arrays for testing
testing_images = np.array(test_x)
testing_labels = np.array(test_y)

# An array containing the possible classifications
class_names = ['drawings', 'engraving', 'iconography', 'painting', 'sculpture']

# A model reconstructed based on the previous model
model = load_model(os.getcwd() + "\\Trained_Model\\model.h5")

# Evaluates the testing samples
test_loss, test_acc = model.evaluate(testing_images, testing_labels, verbose=0)

print("Test accuracy:", test_acc)
print("Test loss:", test_loss)

predictions = model.predict(testing_images)

# View Confusion Matrix
cm = confusion_matrix(testing_labels, np.argmax(predictions, axis=1))
print(cm)