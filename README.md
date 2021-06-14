# Classifying Plant Type by Leaf

## Abstract
The goal of this project was to create a neural network model to classify the type of plant with an image of a plant leaf. 11 different plant types are compatible with the final model. The data was provided by [Mendeley Data](https://data.mendeley.com/datasets/hb74ynkjcn/1). 
I leveraged the power of neural networks and keras image generator to train and build a model with an iterative validation scheme to improve the model to a .93 accuracy for final test scores. There is a steramlit app that streamlines this classification model onto an interactive interface. Once the model identifies the plant leaf, it provides some basic information about that plant!
**This model is deployed through a user friendly application that allows the user to upload their own image of a leaf to have it classified!**
https://github.com/mehiks11/Plant_ID/blob/master/Images/plantappmov.mp4

## Design
This project is based in basic image classification project design built to use the power of neural networks to learn image processing networks and to classify specific plant leaves by plant type. The data is provided by [Mendeley Data](https://data.mendeley.com/datasets/hb74ynkjcn/1). Classifying plant leaves accurately with machine learning models can enable one to identify a plant type. This prototype model can be built upon to include various more plants. Plant identification is vital in promoting environmental interest among the population and in producing an educational beginning to plant owners. 

## Repo Setup
### App Stuff

- [This folder](https://github.com/mehiks11/Plant_ID/tree/master/Images) holds all the images for the output for the app results.
- [This folder](https://github.com/mehiks11/Plant_ID/tree/master/AppStuff) holds all the code and fie that holds uploaded images to run through the app.


## Model Building & Code
- [This notebook](https://github.com/mehiks11/Plant_ID/blob/master/Code/Data%20Management%20--%20Splitting%20data%20(1).ipynb) holds the code for initial data management, including creating folder directories for train/validation/test split.
- [This notebook](https://github.com/mehiks11/Plant_ID/blob/master/Code/Initial%20Modeling%20(1)%20--%20Simple%20Model.ipynb) holds data for intial baseline modeling of a simple Neural network model.
- [This notebook](https://github.com/mehiks11/Plant_ID/blob/master/Code/Modeling%20-%20Baseline%20Model%20%2B%20Modifications%20(1).ipynb) holds the code for further modeling and modifications.
- [This notebook](https://github.com/mehiks11/Plant_ID/blob/master/Code/Final%20Model%20.ipynb) holds the code for the final selected model and four test run images.
- The two "model" files in the [Code](https://github.com/mehiks11/Plant_ID/tree/master/Code) folder hold the final model weights for deployment in the app. 

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

