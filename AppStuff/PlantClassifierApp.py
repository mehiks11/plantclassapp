import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import keras
import numpy as np
from keras.models import model_from_json
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator

##Theme
# [theme]
primaryColor="#bcc986"
backgroundColor="#9de8cc"
secondaryBackgroundColor="#dca9d8"
textColor="#001af9"
font="monospace"

conv_base = VGG16(weights='imagenet',
                  include_top=False)
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

#####

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


####image processing


datagen = ImageDataGenerator(rescale=1./255)
batch_size = 10

def extract_features(directory, sample_count):
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count,1))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='categorical')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size : (i + 1) * batch_size] = features_batch
        labels[i * batch_size : (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    features = np.reshape(features, (1, 4 * 4 * 512))
    return features

#####

###

def teachable_machine_classification(img_dir):
    model = loaded_model
    features = extract_features(img_dir,1)
    predictions = model.predict (features)
    return np.argmax(predictions) # return position of the highest probability

#####


cover_image = Image.open("/Users/mehikapatel/Plant_NN_Project/Images/pomegranate.jpg")
jatophra = Image.open("/Users/mehikapatel/Plant_NN_Project/Images/jatophra.jpg")
pon = Image.open("/Users/mehikapatel/Plant_NN_Project/Images/pongamia.jpg")
pom = Image.open("/Users/mehikapatel/Plant_NN_Project/Images/pomegranate.jpg")
mango = Image.open("/Users/mehikapatel/Plant_NN_Project/Images/mango.jpg")
alstonia = Image.open("/Users/mehikapatel/Plant_NN_Project/Images/alstonia.jpg")
chinar = Image.open("/Users/mehikapatel/Plant_NN_Project/Images/chinar.jpg")
lemon = Image.open("/Users/mehikapatel/Plant_NN_Project/Images/lemon.jpg")
guava = Image.open("/Users/mehikapatel/Plant_NN_Project/Images/guava.jpg")
basil = Image.open("/Users/mehikapatel/Plant_NN_Project/Images/basil.jpg")
jamun = Image.open("/Users/mehikapatel/Plant_NN_Project/Images/jamun.jpg")
arjun = Image.open("/Users/mehikapatel/Plant_NN_Project/Images/arjun.jpg")

st.title("Plant Classifier")
st.image(cover_image,width=200)

st.header("Guess this Plant!")
st.subheader("About This App:")
st.markdown("Hello world. I am an app that is built upon a neural network model trained to classify 11 different plants by an image of a plant leaf! I have the ability to classify the following plant types with 93 percent accuracy. ")
st.markdown("* Jatophra")
st.markdown("* Pongamia Pinnata") 
st.markdown("* Pomegranate") 
st.markdown("* Mango") 
st.markdown("* Alstonia Scholaris") 
st.markdown("* Chinar ") 
st.markdown("* Lemon") 
st.markdown("* Guava") 
st.markdown("* Basil") 
st.markdown("* Jamun") 
st.markdown("* Arjun Plant") 
st.markdown("Follow the directions below to test me!")
st.text("Upload an image of a plant leaf you suspect might be one of the 11 plants above. ")



uploaded_file = st.file_uploader("Upload a clear image(jpg) of a plant leaf against a dark background here!", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image.save("out/image/out.jpg")


dir = "/Users/mehikapatel/Plant_NN_Project/AppStuff/out"

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Plant Leaf', use_column_width=True)
    st.write("Classifying...")
    label = teachable_machine_classification(dir)
    # st.text(label)
    if label == 6   :
        st.markdown("## Jatophra")
        st.markdown("- Plant type: (Jatophra Curcas) Flowering Plant ")
        st.markdown("- Common Name: Coral Plant")
        st.markdown("- Native to: Mexico/Central America")
        st.image(jatophra,width=200)
        st.markdown("**Care Tips:** ")
        st.markdown("   * Well Drained Soil")
        st.markdown("   * Can handle partial sunlight, prefer full sunlight")
        st.markdown("   * Avoid frost zones")
    elif label == 9:
        st.markdown("## Pomegranate")
        st.markdown("- Plant type: Fruit Bearing Deciduous Shrub")
        st.markdown("- Common Name: Pomegranate ")
        st.markdown("- Native to: Mediterranean Region ")
        st.image(pom,width=200)
        st.markdown("** Care Tips:** ")
        st.markdown("   * 50-60 in water/year")
        st.markdown("   * Maintain soil moisture in summer and early fall to prevent fruit splitting")
        st.markdown("   * 4-6 hours of bright sunlight/day (may need to supplement with grow lights)")
    elif label == 8:
        st.markdown("## Mango")
        st.markdown("- Plant type: Fruit Bearing Flowering Tree")
        st.markdown("- Common Name: Mango")
        st.markdown("- Native to: Myanmar, Bangladesh & Northern India")
        st.image(mango,width=200)
        st.markdown("**Care Tips: **")
        st.markdown("   * Above 50% Humidity")
        st.markdown("   * Always above 50 Deg F")
        st.markdown("   * Every day for first 6 weeks, every 2-3 days after")
    elif label == 10:
        st.markdown("## Pongamia Pinnata")
        st.markdown("- Plant type: Tree")
        st.markdown("- Common Name: Karanji (Hindi) ")
        st.markdown("- Native to: Eastern & Tropical Asia, Australia, & Pacific Islands")
        st.image(pon,width=200)
        st.markdown("**Care Tips: **")
        st.markdown("   * Full sun")
        st.markdown("   * Well drained soil")
    elif label == 0:
        st.markdown("## Alstonia Scholaris")
        st.markdown("- Plant type: Evergreen Tropical Tree")
        st.markdown("- Common Name: Blackboard Tree; Devil's Tree ")
        st.markdown("- Native to: Southern China, Tropical Asia, Australasia ")
        st.image(alstonia,width=200)
        st.markdown("**Care Tips: **")
        st.markdown("   * Water regularly, do not overwater")
        st.markdown("   * Sun to partial shade")
    elif label == 3:
        st.markdown("## Chinar")
        st.markdown("- Plant type: Deciduous Tree")
        st.markdown("- Common Name: Old world sycamore; Oriental Plane")
        st.markdown("- Native to: Kashmiri Valley")
        st.image(chinar,width=200)
        st.markdown("**Care Tips: **")
        st.markdown("   * Full to partial sun")
        st.markdown("   * Water once a day")
        st.markdown("   * Drought tolerant")
        st.markdown("   * Rich, moist, well-drained, fertile soil  (alkaline or acidic)")
    elif label == 7:
        st.markdown("## Lemon")
        st.markdown("- Plant type: Evergreen Tree")
        st.markdown("- Native to: South Asia")
        st.image(lemon,width=200)
        st.markdown("**Care Tips: **")
        st.markdown("   * Needs good drainage")
        st.markdown("   * Consistent and regular watering")
        st.markdown("   * Full sun (8-10 hours) with southern exposure")
    elif label == 4:
        st.markdown("## Guava")
        st.markdown("- Plant type: Evergreen Shrub")
        st.markdown("- Common Name: Guava")
        st.markdown("- Native to: Carribean, Central America, South America")
        st.image(guava,width=200)
        st.markdown("**Care Tips: **")
        st.markdown("   * Water frequently ")
        st.markdown("   * Good drainage")
        st.markdown("   * Full sun")
    elif label == 2:
        st.markdown("## Basil")
        st.markdown("- Plant type: Culinary Herb")
        st.markdown("- Common Name: Basil ")
        st.markdown("- Native to: Tropical regions from Central Africa to Southeast Asia")
        st.image(basil,width=200)
        st.markdown("**Care Tips:** ")
        st.markdown("   * Water frequently (1 in/week)")
        st.markdown("   * Fertilize lightly ")
        st.markdown("   * Pinch back leaves")
        st.markdown("   * Lots of sunlight!")
    elif label == 5:
        st.markdown("## Jamun")
        st.markdown("- Plant type: Evergreen Tropical Tree")
        st.markdown("- Common Name: Black Plum")
        st.markdown("- Native to: South & Southeast Asia")
        st.image(jamun,width=200)
        st.markdown("**Care Tips:**" )
        st.markdown("   * Enough water to keep soil moist")
        st.markdown("   * Keep in partial shade")
    elif label == 1:
        st.markdown("## Arjun")
        st.markdown("- Plant type: Terminalia Tree")
        st.markdown("- Common Name: Arjuna ")
        st.markdown("- Native to: India & Sri Lanka")
        st.image(arjun,width=200)
        st.markdown("**Care Tips:** ")
        st.markdown("   * Full Sun")
        st.markdown("   * Well Drained, moderately fertile soil ")
        

