
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

'''Read the data'''
rough_data = np.loadtxt("ex1data2.txt", delimiter=",")
data = np.asmatrix(rough_data)
X_temp = data[:,[0,1]]
y = data[:, 2]
m = y.shape[0]

'''Normalisation'''
mean = X_temp.mean(axis=0)
sigma = np.std(X_temp, axis=0)
X_norm = (X_temp-mean)/sigma

''' Compute the cost function J(\theta) (optional)''' 
theta = np.array([[0],[0],[0]])
X = np.hstack((np.ones((m,1)), X_norm))

M = np.dot(X,theta) - y
J = (0.5/m)*(np.dot(M.T, M))

'''Perform gradient descent'''
alpha = 1

num_iter = 400
for i in range(num_iter):
    M = np.dot(X,theta) - y
    theta = theta - (alpha/m)*(np.dot(X.T, M))

'''Plots'''

'''Setting the grid for the predicted surface'''
x = np.arange(-5, 5, 0.25)
y = np.arange(-5, 5, 0.25)
x, y = np.meshgrid(x,y)
z = theta[0,0] + x*theta[1,0] + y*theta[2,0]
fig = plt.figure()
ax = fig.gca(projection='3d')

'''Plotting the surface and (normalised) data'''
surf = ax.plot_surface(x,y,z)
ax.scatter(X_norm[:,0], X_norm[:,1], rough_data[:,2], c='r')
plt.show()
