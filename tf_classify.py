from keras.models import load_model
import argparse
import cv2
import numpy as np
import os

# Parser that takes the name/location of an image file to be classified by the model
parser = argparse.ArgumentParser(description="Process the name and location of an image")
parser.add_argument('image_name', type=str)
arg = parser.parse_args()

print(arg.image_name)
# An array containing the possible classifications
class_names = ['drawings', 'engraving', 'iconography', 'painting', 'sculpture']

# A model reconstructed based on the previous model
model = load_model(os.getcwd() + "\Trained_Model\model.h5")

# Constructs a numpy array around the provided image
imgs = []
img = cv2.imread(os.getcwd() + "\\" + arg.image_name, 0)
img = cv2.resize(img, (64,64))
imgs.append(img)

# Creates a prediction based on the loaded model
p_img = np.array(imgs) / 255.0
prediction = model.predict(p_img)

# Prints the prediction the model made
print(prediction[0])
print("Predicted the class of this image as: " + class_names[np.argmax(prediction[0])])