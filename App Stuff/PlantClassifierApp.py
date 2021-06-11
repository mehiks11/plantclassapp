import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import keras
import numpy as np
from plantml import teachable_machine_classification

cover_image = Image.open("Images/Pomegranate.jpg")

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
    st.image(image, caption='Uploaded Plant Leaf', use_column_width=True)
    st.write("Classifying...")
    label = teachable_machine_classification(image)
    if label == 1:
        st.md("## Jatophra \
            - Plant type: (Jatophra Curcas) Flowering Plant \
            - Common Name: Coral Plant\
            - Native to: Mexico/Central America\
            - Image of plant: ![Jatophra](Images/jatophra.jpg)\
            - Care Tips: \
            * Well Drained Soil\
            * Can handle partial sunlight, prefer full sunlight\
            * Avoid frost zones")
    elif label == 2:
        st.md("## Pomegranate\
            - Plant type: Fruit Bearing Deciduous Shrub\
            - Common Name: Pomegranate \
            - Native to: Mediterranean Region \
            - Image of plant: ![Pomegranate](Images/Pomegranate.jpg)\
            - Care Tips: \
            * 50-60 in water/year\
            * Maintain soil moisture in summer and early fall to prevent fruit splitting\
            * 4-6 hours of bright sunlight/day (may need to supplement with grow lights)")
    elif label == 3:
        st.md("## Mango\
            - Plant type: Fruit Bearing Flowering Tree\
            - Common Name: Mango\
            - Native to: Myanmar, Bangladesh & Northern India\
            - Image of plant: ![Mango](Images/Mango.jpg)\
            - Care Tips: \
            * Above 50% Humidity\
            * Always above 50 Deg F\
            * Every day for first 6 weeks, every 2-3 days after")
    elif label == 4:
        st.md("## Pongamia Pinnata\
            - Plant type: Tree \
            - Common Name: Karanji (Hindi) \
            - Native to: Eastern & Tropical Asia, Australia, & Pacific Islands\
            - Image of plant: ![Pongamia Pinnata](Images/Pongamia.jpg)\
            - Care Tips: \
            * Full sun\
            * Well drained soil")
    elif label == 5:
        st.md("## Alstonia Scholaris\
            - Plant type: Evergreen Tropical Tree\
            - Common Name: Blackboard Tree; Devil's Tree \
            - Native to: Southern China, Tropical Asia, Australasia \
            - Image of plant: ![Alstonia Scholaris](Images/Alstonia.jpg)\
            - Care Tips: \
            * Water regularly, do not overwater\
            * Sun to partial shade")
    elif label == 6:
        st.md("## Chinar\
            - Plant type: Deciduous Tree\
            - Common Name: Old world sycamore; Oriental Plane\
            - Native to: Kashmiri Valley\
            - Image of plant: ![Chinar](Images/Chinar.jpg)\
            - Care Tips: \
            * Full to partial sun\
            * Water once a day\
            * Drought tolerant\
            * Rich, moist, well-drained, fertile soil  (alkaline or acidic)")
    elif label == 7:
        st.md("## Lemon\
            - Plant type: Evergreen Tree\
            - Native to: South Asia\
            - Image of plant: ![Lemon](Images/Lemon.jpg)\
            - Care Tips: \
            * Needs good drainage\
            * Consistent and regular watering\
            * Full sun (8-10 hours) with southern exposure")
    elif label == 8:
        st.md("## Guava\
            - Plant type: Evergreen Shrub\
            - Common Name: Guava\
            - Native to: Carribean, Central America, South America\
            - Image of plant: ![Guava](Images/Guava.jpg)\
            - Care Tips: \
            * Water frequently \
            * Good drainage\
            * Full sun")
    elif label == 9:
        st.md("## Basil\
            - Plant type: Culinary Herb \
            - Common Name: Basil \
            - Native to: Tropical regions from Central Africa to Southeast Asia\
            - Image of plant: ![Basil](Images/Basil.jpg)\
            - Care Tips: \
            * Water frequently (1 in/week)\
            * Fertilize lightly \
            * Pinch back leaves\
            * Lots of sunlight!")
    elif label == 10:
        st.md("## Jamun\
            - Plant type: Evergreen Tropical Tree \
            - Common Name: Black Plum\
            - Native to: South & Southeast Asia\
            - Image of plant: ![Jamun](Images/Jamun.jpg)\
            - Care Tips: \
            * Enough water to keep soil moist\
            * Keep in partial shade")
    elif label == 11:
        st.md("## Arjun\
        - Plant type: Terminalia Tree\
        - Common Name: Arjuna \
        - Native to: India & Sri Lanka\
        - Image of plant: ![Arjun](Images/Arjun.jpg)\
        - Care Tips: \
        * Full Sun\
        * Well Drained, moderately fertile soil ")
        

