
#simple image-recognition algorithm that can correctly classify pictures as cat or non-cat.

import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from public_tests import *


#TO LOAD DATASET

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


# Find the values for:
# m_train (number of training examples)
# m_test (number of test examples)
# num_px (= height = width of a training image)

# Remember that `train_set_x_orig` is a numpy-array of shape (m_train, num_px, num_px, 3). For instance, you can access `m_train` by writing `train_set_x_orig.shape[0]`.
# a training set of m_train images labeled as cat (y=1) or non-cat (y=0)

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

# print ("Number of training examples: m_train = " + str(m_train))
# print ("Number of testing examples: m_test = " + str(m_test))
# print ("Height/Width of each image: num_px = " + str(num_px))
# print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
# print ("\n")

#to reshape images of shape (num_px, num_px, 3) in a numpy-array of shape (num_px $*$ num_px $*$ 3, 1)

train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test,-1).T

#NOTE:the pixel value is actually a vector of three numbers ranging from 0 to 255.

# #One common preprocessing step in machine learning is to center and standardize your dataset, meaning that you substract 
# the mean of the whole numpy array from each example, and then divide each example by the standard deviation of the whole 
# numpy array.
# But for picture datasets, it is simpler and more convenient and works almost as well to just divide every row 
# of the dataset by 255 (the maximum value of a pixel channel).

# to standardize our dataset.

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

# to design a simple algorithm to distinguish cat images from non-cat images.
# for this purpose we will use "logistic Regression model".
# this algorithm is mostly use for binary classification  which predict probabilites for binary classification problem.
#Logistic Regression is actually a very simple Neural Network!

# steps
# The main steps for building a Neural Network are:
# 1. Define the model structure (such as number of input features) 
# 2. Initialize the model's parameters
# 3. Loop:
#     - Calculate current loss (forward propagation)
#     - Calculate current gradient (backward propagation)
#     - Update parameters (gradient descent)

# let first implement sigmoid funtion 

def sigmoid(z):
    
    x = 1 / (1 + np.exp(-z))
    return x

#let Initialized parameters
#This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

def initialize_with_zeros(dim):
   # Argument:
   #dim -- size of the w vector we want (or number of parameters in this case)
      
    w = np.zeros((dim,1))
    b = 0.0
    
    return w,b

 # Now as parameters are initialized, lets implement "forward" and "backward" propagation steps for learning the parameters.
 # first Implement a function `propagate()` that computes the cost function and its gradient.

def propagate(w, b, X, Y):
    
    m = X.shape[1]
    i = np.dot(w.T, X) + b
    A =  sigmoid(i)
    cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A) )
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
   
    dw = (1 / m) * np.dot(X,((A - Y).T))
    db = (1 / m) * np.sum(A - Y)
    cost = np.squeeze(np.array(cost))
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

#optimization funtion

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):

#     This function optimizes w and b by running a gradient descent algorithm
#     Tips:
#     You basically need to write down two steps and iterate through them:
#         1) Calculate the cost and the gradient for the current parameters. Use propagate().
#         2) Update the parameters using gradient descent rule for w and b.
    
    w = copy.deepcopy(w)
    b = copy.deepcopy(b)
    costs = []
    
    for i in range(num_iterations):
       
        grads, cost = propagate(w, b, X, Y)
       
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update w and b
   
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
          
            # Print the cost every 100 training iterations
            if print_cost:
                print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

#lets define "predict function
# The previous function will output the learned w and b. We are able to use w and b to predict the labels for a dataset X
# steps
# 1. Calculate = A = sigma(w^T X + b)
# 2. Convert the entries of A into 0 (if activation <= 0.5) or 1 (if activation > 0.5), stores the predictions in a vector `Y_prediction`. If you wish, you can use an `if`/`else` statement in a `for` loop (though there is also a way to vectorize this). 

def predict(w,b,X):
    #note : X -- data of size (num_px * num_px * 3, number of examples)
    #Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    
    #NOW Compute vector "A" predicting the probabilities of a cat being present in the picture
    
    A = sigmoid(np.dot(w.T,X) + b)
    
    for i in range(A.shape[1]):
        
        # TO Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0,i] <= 0.5:
            Y_prediction[0,i] = 0
        else:
             Y_prediction[0,i] = 1
            
    return  Y_prediction


#NOW
#Lets Merge all functions into a model
#note that
#       Y_prediction_test for your predictions on the test set
#     - Y_prediction_train for your predictions on the train set
#     - parameters, grads, costs for the outputs of optimize()

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    
    #lets start
    #step 1:
    # initialize parameters with zeros
    # and use the "shape" function to get the first dimension of X_train
         
        dim = X_train.shape[0]
        w , b = initialize_with_zeros(dim)
    
    #step 2:
    #here we need 
    # Gradient descent 
    # params, grads, costs 
    
        #which is return by optimize funtion so here we will call it
    
        params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost=False)
    
        w = params["w"]
        b = params["b"] 
    
        #Now to get the prediction values i.e
        Y_prediction_test = predict(w,b,X_test)
        Y_prediction_train = predict(w,b,X_train)
         
        #Print train/test Errors
        if print_cost:
            
              #SECOND
            print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
            print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
            print("\n")
       
        d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    
        return d


    
#NOW Finally we have our  model implemention has been complete
#lets train our model

Logistic_regression_model = model(train_set_x, train_set_y, test_set_x,test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

#FIRST

# List of learning rates to try
learning_rates = [0.01, 0.001, 0.0001]

# Dictionary to store trained models
models = {}

# Train models with different learning rates
# for lr in learning_rates:
#     print("Training a model with learning rate: " + str(lr))
#     models[str(lr)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=lr, print_cost=False)
#     print('\n' + "-------------------------------------------------------" + '\n')

# # Plot costs of models
# for lr in learning_rates:
#     plt.plot(np.squeeze(models[str(lr)]["costs"]), label=str(models[str(lr)]["learning_rate"]))

# plt.ylabel('Cost')
# plt.xlabel('Iterations (hundreds)')

# legend = plt.legend(loc='upper center', shadow=True)
# frame = legend.get_frame()
# frame.set_facecolor('0.90')
# plt.show()
# print("\n")

#THIRD

# lets Visualize predictions 
# You can choose any image index to visualize its prediction

import random
index = random.randint(1,50)
plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
print("Predicted: image of " + classes[int(Logistic_regression_model['Y_prediction_test'][0, index])].decode("utf-8"))


# In[ ]:





# In[ ]:




