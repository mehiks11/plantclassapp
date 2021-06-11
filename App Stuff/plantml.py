import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import keras
import numpy as np
def build_model():
    model = keras.Sequential([
    keras.layers.InputLayer(input_shape=(8192,)),
    keras.layers.Dense(units=150, activation="relu"),
    keras.layers.Dense(units=125, activation="relu"),
    keras.layers.Dense(units=100, activation="relu"),
    keras.layers.Dense(units=75, activation="relu"),
    keras.layers.Dense(units=50, activation="relu"),
    keras.layers.Dense(units=25, activation="relu"),
    keras.layers.Dense(units=11, activation= "softmax"),
])
    model.compile("nadam", loss="categorical_crossentropy", metrics=["acc"])
    return model

model2 = build_model()
model2.load_weights("/Users/mehikapatel/Plant_NN_Project/Code/model.h5")

def teachable_machine_classification(img):
    model = model2
    #keras.models.load_weights('/Users/mehikapatel/Plant_NN_Project/Code/model.h5')
    data = np.ndarray(shape=(1, 4, 4, 512), dtype=np.float32)
    image = img
    size = (4, 4)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    #turn the image into a numpy array
    image_array = np.asarray(image)
    #flatten image
    image_array = np.reshape(image_array, (1, 4 * 4 * 512))
    #predict
    prediction = model.predict(image_array)
    return np.argmax(prediction) # return position of the highest probability