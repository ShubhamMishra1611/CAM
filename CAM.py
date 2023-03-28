from __future__ import print_function, division
from builtins import range, input

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

from glob import glob

# get the image
img_path = glob('images/*.jpg')

# sanity check
img_path
plt.imshow(image.load_img(np.random.choice(img_path)))
plt.show()

# loading the model
resnet = ResNet50(input_shape = (224,224,3), weights="imagenet", include_top = True)

#sanity check
resnet.summary()



# getting the output before flatten layer
activation_layer = resnet.get_layer('activation_49')
dense_final = resnet.get_layer('fc1000')
W = dense_final.get_weights()[0]


model = Model(inputs = resnet.input, outputs = activation_layer.output)


while True:
    img = image.load_img(np.random.choice(img_path), target_size = (224,224))
    x = preprocess_input(np.expand_dims(img, axis = 0))
    fmaps = model.predict(x)[0]

    probs = resnet.predict(x)
    class_names = decode_predictions(probs)[0]
    print(class_names)
    class_name = class_names[0][1]
    pred = np.argmax(probs[0])

    # getting the weights of the class
    W = W[:,pred]
    
    # now the dot product 
    cam = fmaps.dot(W)

    cam = sp.ndimage.zoom(cam, (32,32), order = 1)

    plt.subplot(1,2,1)# overlay the cam with the image
    plt.imshow(img, alpha=0.5)
    plt.imshow(cam, cmap = 'jet', alpha = 0.4)
