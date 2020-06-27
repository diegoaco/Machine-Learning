''' IMPLEMENTATION OF A ONE-VARIABLE LINEAR REGRESSION ALGORITHM USING GRADIETN DESCENT '''

import numpy as np
from matplotlib import pyplot as plt

'''Read the data'''
data1 = np.loadtxt("ex1data1.txt", delimiter=",")
data = np.asmatrix(data1) # This convert the data into matrix form
X_temp = data[:,0]
y = data[:,1]

''' Compute the cost function J(\theta)'''
m = X_temp.shape[0]
theta = np.array([[-1],[2]])

ones = np.ones((m,1))
X = np.hstack((ones, X_temp))

M = np.dot(X,theta) - y
J = (0.5/m)*(np.dot(M.T, M))

'''Perform gradient descent'''
alpha = 0.01
num_iter = 1500
for i in range(num_iter):
    M = np.dot(X,theta) - y
    theta = theta - (alpha/m)*(np.dot(X.T, M))
print(theta)

'''Prediction'''
pred = np.dot(X,theta)

'''Plot the data and the line'''
plt.scatter(data1[:,0], data1[:,1])
plt.plot(data1[:,0],pred, c = 'r')
plt.show()
