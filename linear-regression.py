''' Implementation of Machine Learning gradient descent algorithm,
compared to a normal equation approach '''


import numpy as np
from matplotlib import pyplot as plt

'''Read the data'''
data1 = np.loadtxt("/Users/diego/python/Machine Learning/ex1data1.txt", delimiter=",")

# Convert the data into matrix form
data = np.asmatrix(data1)
X_temp = data[:,0]
y = data[:,1]
m = X_temp.shape[0]  # Number of training examples

# Construct the matrix X:
ones = np.ones((m,1))
X = np.hstack((ones, X_temp))

''' Using the normal equation '''
A = np.linalg.inv(np.dot(X.T, X))
B = np.dot(A, X.T)
theta_nor = np.dot(B, y)

# Calculate the cost function for theta_nor 
M = np.dot(X, theta_nor) - y
cost_nor = (0.5/m)*(np.dot(M.T, M))

print('Using the normal equation:')
print('The parameters are:', theta_nor[0,0], theta_nor[1,0])
print('The cost function minimum is:', cost_nor[0,0])

''' Perform gradient descent '''
alpha = 0.01   # learning rate
i_max = 10000  # Max number of steps
cost = [np.inf]   # Cost function vector
e = 1e-10   # difference parameter
theta_grad = np.array([[-1],[1]])  # Ansatz for theta

for i in range(1, i_max+1):
    M = np.dot(X, theta_grad) - y
    theta_grad = theta_grad - (alpha/m)*(np.dot(X.T, M))
    J = (0.5/m)*(np.dot(M.T, M))  # Computes the cost function
    cost.append(J[0,0])
    # Stop if the cost function changes less than e
    if abs(cost[-1] - cost[-2]) < e:
        i_max = i
        break

print('- - - - - - - - - - - - - - - - -\nUsing gradient descent:')
print('The parameters are:', theta_grad[0,0], 'and', theta_grad[1,0])
print('Cost function minimum:', cost[-1])
print('Number of steps:', i_max)

''' Plots '''
# Prediction models:
pred_nor = np.dot(X, theta_nor)  
pred_grad = np.dot(X, theta_grad)
t = [i for i in range(1, i_max+1)]  # Steps vector

# Plot the data and the line
plt.scatter(data1[:,0], data1[:,1])
plt.plot(data1[:,0], pred_nor, c = 'g')
plt.plot(data1[:,0], pred_grad, c = 'r')
plt.xlabel('x-value')
plt.ylabel('y-value')
plt.show()
plt.plot(t, cost[1:])
plt.xlabel('Steps')
plt.ylabel('Cost function')
plt.show()
