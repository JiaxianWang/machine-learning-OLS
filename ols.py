#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# importing necessary libraries
import matplotlib.pyplot as plt
import numpy as np


# In[ ]:


# Solves the Least Squares problem for given X, Y. 
# alpha is the regularization coefficient
# Output is the estimated y for both X and X_all
def solve_ols(X_train, Y_train, X_test, alpha):
    W = np.dot(np.linalg.pinv(np.dot(X_train.T, X_train) + alpha*np.identity(np.shape(X_train)[1])), np.dot(X_train.T, Y_train))
    print ("Optimal W is ", W.flatten())
    return np.dot(X_train, W), np.dot(X_test, W)


# In[ ]:


def run_ols(X_train, Y_train, X_test, Y_test, alpha, plot_X_train, plot_X_test, description):
    Y_LS_train, Y_LS_test = solve_ols(X_train, Y_train, X_test, alpha)
    
    fig, ax = plt.subplots(figsize=(12,12), dpi=50)
    
    # Plotting the train data
    #ax.plot(X_train, Y_train, 'rx')
    # Plotting the prediction of our model on the train data
    #ax.plot(X_train, Y_LS_train, 'bo')

    # Plotting the actual y values for the test data
    ax.plot(plot_X_test, Y_test, 'rx', label='The actual y values for test data')
    # Plotting the prediction of our model on the test data
    ax.plot(plot_X_test, Y_LS_test, 'bo', label='The predicted y values for test data')
    
    ax.legend(loc='lower right', prop={'size': 20})
    ax.set(xlabel='X', ylabel='Y', title=description)
    ax.grid()
    plt.show()
    
    print ("Mean Squarred Error (MSE) of train data: " , np.square(np.linalg.norm(Y_LS_train-Y_train))/Y_train.size)
    print ("Mean Squarred Error (MSE) of test data: " , np.square(np.linalg.norm(Y_LS_test-Y_test))/Y_test.size)


# In[ ]:


# generate n data points based on a combination of sinosuidal and polynomial functions
def generate_data(n):
    X = np.random.rand(n, 1)*5
    X = np.sort(X, axis=0)
    Y = 10 + (X-0.1)*(X-0.1)*(X-0.1) - 5*(X-0.5)*(X-0.5) + 10*X + 5* np.sin(5*X)
    # Adding noise
    Y = Y + 0.1*np.random.randn(n,1)
    return X, Y


# In[ ]:


# Number of training and test points.
n_train = 30
n_test = 1000

# This will be used later for regularization. For now it is set to 0.
alpha = 0

# Generating train and test data.
X_train, Y_train = generate_data(n_train)
X_test, Y_test = generate_data(n_test)

# Homogenous line/hyperplane (goes through the origin)
run_ols(X_train, Y_train, X_test, Y_test, alpha, X_train, X_test, "Homogenous line")

# Non-homogenous line/hyperplane
# First we augment the data with an all 1 column/feature
X_augmented_train = np.concatenate((X_train, np.ones((n_train, 1))), axis=1)
X_augmented_test = np.concatenate((X_test, np.ones((n_test, 1))), axis=1)

# Now we run OLS on the augmented data.
run_ols(X_augmented_train, Y_train, X_augmented_test, Y_test, alpha, X_train, X_test, "Nonhomogenous line")


# In[ ]:





# In[ ]:




