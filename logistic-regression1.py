#IMPLEMENTATION OF A CLASSIFICATION ALGORITHM USING LOGISTIC REGRESSION AND NORMALISATION
#** NOTE: Gradient descent is manually implemented instead of using a built-in tool**

import numpy as np
import math
from matplotlib import pyplot as plt

'''Read data from file and define matrices'''
rough_data = np.loadtxt("ex2data1.txt", delimiter=",")
data = np.asmatrix(rough_data)
X_temp = data[:,[0,1]]
y = data[:, 2]
m, n = X_temp.shape[0], X_temp.shape[1]

'''Normalisation function'''
def norm(X_temp):
    mean = X_temp.mean(axis=0)
    sigma = np.std(X_temp, axis=0)
    X_norm = (X_temp-mean)/sigma
    X = np.hstack((np.ones((m,1)), X_norm))
    return X

X_norm = norm(rough_data[:, [0,1]])

'''Sigmoid function'''
def sigmoid(z):
    return 1./(1+np.exp(-z))

'''Cost function and gradient''' 
theta_0 = np.zeros((n+1, 1)) # Initial theta

def cost(theta, X, y):
    h = sigmoid(X@theta)
    id = np.ones((m,1))
    J = (-1./m)*(y.T@np.log(h) + (id - y).T@np.log(id-h))
    grad = (1./m)*(X.T@(h-y))
    return grad

'''Gradient descent (main function)'''
def grad_des(theta, X, y, alpha, num_iter):
    for i in range(num_iter):
        grad = cost(theta, X, y) 
        theta = theta - (alpha*grad)
    return theta

theta = grad_des(theta_0, X_norm, y, 1, 400) # Optimsed value of theta


'''Calculate accuracy'''
p = np.zeros((m,1))
def predictions(theta, X):
    for i in range(m):
        pred = sigmoid(X@theta)
        if pred[i] >= 0.5: p[i] = 1
        else: p[i]= 0 
    return p

p = predictions(theta, X_norm)
print(sum(p == y)[0],"%")


'''Plot the normlaised data and decision boundary'''

x = np.arange(-2, 2, 0.25)
boundary = (-theta[0,0] - x*theta[1,0])/theta[2,0]

#Classify positive and negative cases
pos = np.argwhere(rough_data[:,2] == 1)
neg = np.argwhere(rough_data[:,2] == 0)

plt.scatter(X_norm[pos,1], X_norm[pos,2], c='b')
plt.scatter(X_norm[neg,1], X_norm[neg,2], c='r')
plt.plot(x, boundary, c = 'black')
plt.show()

'''#Find the probability of 'pass' of a particular case
z = np.array([45,85])
mean = X_temp.mean(axis=0)
sigma = np.std(X_temp, axis=0)
z_norm = np.append(1, (z-mean)/sigma )
print(sigmoid(z_norm@theta))'''
