#      Kaggle: Dog Breed Identification        #


This is my submission for the Kaggle Playground Competition, "Dog Breed Identificaiton" (https://www.kaggle.com/c/dog-breed-identification).

Here I am using a Convolutional Neural Network (CNN) implemented in Tensorflow.

The CNN has two convolutional layers and one fully connected layer.  

The data set includes a training and a testing data set.  The training set contains 10,222 images from ImageNet of dogs from 120 different breeds.  To train the network, 9600 images were randomly selected from the training set for optimization of the neural network, and the remaining 622 images were used for validation.  After the network is optimized and validated, the testing set is run through the CNN for submission to Kaggle. 

**This is a work in progress.**

Next steps:

-Optimize the learning rate to obtain the best model fit vs. training time

-Implement GPU processing to significantly speed up network training

-Examine how resizing images distorts features and determine if resized image resolution needs to be increased

-More updates to come


