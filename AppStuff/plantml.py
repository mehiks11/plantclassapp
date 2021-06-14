import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import keras
import numpy as np
from keras.models import model_from_json

# load json and create model
json_file = open('/Users/mehikapatel/Plant_NN_Project/Code/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/Users/mehikapatel/Plant_NN_Project/Code/model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile("nadam", loss="categorical_crossentropy", metrics=["acc"])

def teachable_machine_classification(img):
    model = loaded_model
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
# load json and create model
json_file = open('/Users/mehikapatel/Plant_NN_Project/Code/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("/Users/mehikapatel/Plant_NN_Project/Code/model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile("nadam", loss="categorical_crossentropy", metrics=["acc"])

def teachable_machine_classification(img):
    model = loaded_model
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