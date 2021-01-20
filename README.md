# Machine Learning - Art Style Classification
An experiment intended to attempt the machine learning classification of images composed of a variety of different art styles.

The entirety of this project took place across 2018, beginning with the inital learning process of the fundamentals of machine learning, how it works, how to approach
and code it, and advancing into the design and development of deep learning convolutional neural networks. Contained within this project are a set of scripts for
training, testing, and classifying images of art, along with the specific algorithmic layers of a deep learning neural network I designed with the Keras Python framework.

Also included is a presentation detailing the specific results obtained from this deep learning convolutional neural network. If you're only interested in the outcome
of this research this presentation should largely give you an appropriate overview of the reasoning, process, and results from this approach.

# The Dataset
The dataset implemented within this project was found off Kaggle and can be accessed through the below link.
https://www.kaggle.com/thedownhill/art-images-drawings-painting-sculpture-engraving/home

(Disclaimer: During training, it was discovered
that images organized within the drawing section of this dataset were incorrectly split between the separate training and testing datasets, producing slightly less
accurate results. For a more accurate model, it is recommended to remove images found to be 100% accurate from the testing dataset and retraining the model)
