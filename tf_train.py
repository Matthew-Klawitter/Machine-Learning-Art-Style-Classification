from PIL import Image
from scipy import misc
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import os

classes = os.listdir("Images")
x = [] # An array that holds image data
y = [] # An array that holds the class an image belongs to

for image_class in classes:
    imgs = os.listdir("Images\\" + image_class)

    for image in imgs:
        img = misc.imread("Images\\" + image_class + "\\" + image)
        img = misc.imresize(img, (64, 64))
        x.append(img)
        y.append(image_class)

# The final organized, numpy datasets
X_train = np.array(x) # Issue here
Y_train = np.array(y)

# Creates a Sequential tensorflow neural network under the keras framework
model = Sequential()

# Adding layers to the model
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

# Compiles the model together, the last step to establishing the neural network
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=10, batch_size=32, shuffle=True)
scores = model.evaluate(X_train, Y_train, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Serializes the model to json format
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Serializes the weights to HDF5 format
model.save_weights("model.h5")
print("Success! Model has been saved to disk.")