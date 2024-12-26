# MultiLayerPerceptron-CNN
Project Emphasizes on Deep Learning with CNN Algorithms. Problem Statement can be found in the ReadME file.

This project contains two tasks. Task 1 is to implement a three-layer MLP from 
scratch and then train and test it with the given data. Task 2 is to experiment with “deep” 
learning on the CIFAR-10 dataset.  
Task 1 specific requirements:
1. Implement a 2-nH-1 MLP from scratch, without using any built-in library. Write your 
code in such a way that you may try different number of hidden nodes easily. Pick your 
own activation function; use MSE error as the loss. Decide on other details such as 
batch/mini-batch/stochastic learning, learning rate, whether to use momentum term, 
etc. 
2. Let nH take the following values: 2, 4, 6, 8, 10. For each case, train the network with 
the first 1500 training samples from each class, given in train_class0.mat (for class 0) 
and train_class1.mat (for class 1), respectively. In the data files, each line is a 2-D 
sample, and the number of lines is the number samples in that class. We have 2000 
training samples for each class. You will use only the first 1500 training samples from 
each class for training the network and reserve the remaining 500 (1000 total for 2 
class) as the “validation set”. Train the network until the learning loss/error (J or J/n as 
defined in the lecture slides) for the validation set no longer decreases. Then test the 
network on the testing data, given in test_class0.mat (for class 0) and test_class1.mat 
(for class 1), respectively. We have 1000 testing samples for each class.  
Note: you may want to do “feature normalization”, as done in Project Part 1 (i.e., using 
the training data to estimate the mean and STD, and then use them for normalizing all 
the data before any further processing). 
3. Plot the learning curves (for the training set, validation set, and the test set, 
respectively), for each value nH.   
4. Report at which value of nH, your network gives the best classification accuracy for the 
testing set. (Note: if you start learning from different random initializations, you might 
have different results even for a given nH. So a more meaningful answer to this question 
would require you to try different initializations and then take the average.)

Task 2 specific requirements: 
In this task, we will perform the classification task, using a convolutional neural network. 
The dataset is the CIFAR-10 dataset. We will experiment with a convolutional neural 
network with the following parameter settings:  - The input size is the size of the image (32x32x3).  
1 
- First layer – Convolution layer with 32 kernels of size 3x3. It is followed by ReLU 
activation layer and batch normalization layer. - Second layer – Convolution layer with 32 kernels of size 3x3. It is followed by ReLU 
activation layer and batch normalization layer. - Third layer – Max pooling layer with 2x2 kernel.  - Fourth layer – Dropout layer with 0.2 probability of setting a node to 0 during training.  - Fifth layer – Convolution layer with 64 kernels of size 3x3. It is followed by ReLU 
activation layer and batch normalization layer. - Sixth layer – Convolution layer with 64 kernels of size 3x3. It is followed by ReLU 
activation layer and batch normalization layer. - Seventh layer – Max pooling layer with 2x2 kernel.  - Eighth layer – Dropout layer with 0.3 probability of setting a node to 0 during training.  - Ninth layer – Convolution layer with 128 kernels of size 3x3. It is followed by ReLU 
activation layer and batch normalization layer. - Tenth layer – Convolution layer with 128 kernels of size 3x3. It is followed by ReLU 
activation layer and batch normalization layer. - Eleventh layer – Max pooling layer with 2x2 kernel.  - Twelfth layer – Dropout layer with 0.4 probability of setting a node to 0 during training.  - Last layer – Fully connected layer with 10 nodes (corresponding to the 10 classes) and 
SoftMax activation function. 
The padding for the layers is set as ‘same’ which means zero padding is added to keep 
the output dimensions same as input for the layer. Use the following code snippet to 
download the CIFAR-10 dataset:  
from keras.datasets import cifar10 
(x_train, y_train), (x_test, y_test) = cifar10.load_data() 
You will need to use Keras deep learning library to implement the network. You will train 
the network with the training set and then test it on the testing set.  
You are also required to experiment with the code by changing some of the hyper
parameters and then analyze its effect on the test accuracy. Change the following 
settings and report the test accuracy:  - Change learning rate to – i) 0.05 ii) 0.0001 - Change kernel size for first convolutional layer to 7x7 - Remove all the batch normalization layers in the network  - Change batch size – i) 16 ii) 256 
