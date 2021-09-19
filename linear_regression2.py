''' Multivariable machine learning implementation using gradient descent and
the normal equation, both normalised. Also computes the R^2 value and tracks the
behaviour of the cost function. '''

import numpy as np
from matplotlib import pyplot as plt

'''Read the data'''
rough_data = np.loadtxt("/Users/diego/python/Machine Learning/ex1data2.txt", delimiter=",")
data = np.asmatrix(rough_data)
X_temp = data[:,[0,1]]
y_temp = data[:, 2]
m = y_temp.shape[0]

'''Normalisation and construction of the X matrix'''
X_mean = np.mean(X_temp, axis=0)
X_sigma = np.std(X_temp, axis=0)
X_nor = (X_temp-X_mean)/X_sigma
X = np.hstack((np.ones((m,1)), X_nor))
y_mean = np.mean(y_temp, axis=0)
y_sigma = np.std(y_temp, axis = 0)
y = (y_temp - y_mean)/y_sigma


''' Using the normal equation '''
A = np.linalg.inv(np.dot(X.T, X))
B = B = np.dot(A, X.T)
theta_nor = np.dot(B, y)

# Compute the cost function for theta_nor (optional)
M = np.dot(X, theta_nor) - y
cost_nor = (0.5/m)*(np.dot(M.T, M))

print('Using the normal equation:')
print('The parameters are:', theta_nor[0,0], theta_nor[1,0], 'and', theta_nor[2,0])
print('Cost function minimum is:', cost_nor[0,0])


'''Perform gradient descent'''
alpha = 0.01   # learning rate
i_max = 5000  # Max number of steps
cost = [np.inf]   # Cost function vector
e = 1e-10   # difference parameter
theta_grad = np.array([[0],[0], [0]])  # Ansatz for theta

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
print('The parameters are:', theta_grad[0,0], theta_grad[1,0], 'and', theta_grad[2,0])
print('Cost function minimum:', cost[-1])
print('Number of steps:', i_max)


''' R^2 analysis '''
# Prediction models:
pred_nor = np.dot(X, theta_nor)
pred_grad = np.dot(X, theta_grad)

# R^2 calculation:
res = pred_grad - y
ss_res = np.dot(res.T, res)
res_var = y - np.mean(y, axis=0)
ss_tot = np.dot(res_var.T, res_var)
r2 = 1 - (ss_res/ss_tot)
print('The R^2 value is:', r2[0,0])


''' Plots '''
# Setting up the grid for the predicted surface
x_range = np.arange(-5, 5, 0.25)
y_range = np.arange(-5, 5, 0.25)
x_coord, y_coord = np.meshgrid(x_range, y_range)
z_coord = theta_nor[0,0] + x_coord*theta_nor[1,0] + y_coord*theta_nor[2,0]
fig = plt.figure()

# Plotting the surface and (normalised) data
ax = fig.gca(projection = '3d')
surf = ax.plot_surface(x_coord, y_coord, z_coord)
ax.scatter(X_nor[:,0], X_nor[:,1], y[:,0], c='r')
plt.show()

# Plot the cost function:
t = [i for i in range(1, i_max+1)]  # Steps vector
plt.plot(t, cost[1:])
plt.xlabel('Steps')
plt.ylabel('Cost function')
plt.show()
