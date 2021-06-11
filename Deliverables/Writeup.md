# Classifying Plant Type by Leaf
Mehika Patel

## Abstract
The goal of this project was to create a neural network model to classify the type of plant with an image of a plant leaf. 11 different plant types are compatible with the final model. The data was provided by [Mendeley Data](https://data.mendeley.com/datasets/hb74ynkjcn/1). 
I leveraged the power of neural networks and keras image generator to train and build a model with an iterative validation scheme to improve the model to a .93 accuracy for final test scores. There is a steramlit app that streamlines this classification model onto an interactive interface. Once the model identifies the plant leaf, it provides some basic information about that plant!

## Design
This project is based in basic image classification project design built to use the power of neural networks to learn image processing networks and to classify specific plant leaves by plant type. The data is provided by [Mendeley Data](https://data.mendeley.com/datasets/hb74ynkjcn/1). Classifying plant leaves accurately with machine learning models can enable one to identify a plant type. This prototype model can be built upon to include various more plants. Plant identification is vital in promoting environmental interest among the population and in producing an educational beginning to plant owners. 

## Data
The dataset contains 2273 images of 11 different plant types. Specifically, the plant types are as follows:
1. Jatropha
2. Pomegranate
3. Mango
4. Pongamia Pinnata
5. Alstonia Scholaris
6. Chinar
7. Lemon
8. Guava
9. Basil 
10. Jamun
11. Arjun Plant

## Algorithms
Algorithsm used included building upon a VGG16 convolutional network, leveraging models built on imagenet. A model with 6 dense hidden layers plus an output softmax layer was added. Several models were experimented with, as can be followed in the [model modification notebook] ().

*Model Evaluation and Selection*
  
The final model resulted in a .93 accuracy score for the test data. 

## Tools
- Numpy for data manipulation
- Keras for model building and image generation
- tensorflow for object manipulation
- Streamlit for  app building

## Communication
In addition to the slides and visuals presented, a streamlit app with plant care reccomendations will be provided.

