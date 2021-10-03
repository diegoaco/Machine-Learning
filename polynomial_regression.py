'''Polynomial regression using a linear algorithm with gradient descent'''

import numpy as np
import matplotlib.pyplot as plt

# CREATE ARTIFICIAL DATA
x_data = 2*np.random.normal(0, 1, 100)
y_data = 2 + 1*x_data -5*(x_data ** 2) + 3*(x_data**3)+ 10*np.random.normal(0, 3, 100)

# DEFINE THE RELEVANT OBJECTS
m = len(x_data)
X_temp = np.asmatrix(x_data).T
X_sq = np.square(X_temp)
X_cube = np.power(X_temp, 3)
y = np.asmatrix(y_data.T)
X = np.hstack((np.ones((m,1)), X_temp, X_sq, X_cube))

# GRADIENT DESCENT
alpha = 0.001   # learning rate
i_max = 100 # Max number of steps
cost = []
theta = np.array([[0],[0],[0],[0]])  # Ansatz for theta

for i in range(1, i_max+1):
    M = np.dot(X, theta) - y.T
    theta = theta - (alpha/m)*(np.dot(X.T, M))
    J = (0.5/m)*(np.dot(M.T, M))  # Computes the cost function
    cost.append(J[0,0])

print('The parameters vector is:', theta)

# R^S ANALYSIS
res = np.dot(X, theta) - y.T
ss_res = np.dot(res.T, res)
res_var = y.T - y.T.mean(axis=0)
ss_tot = np.dot(res_var.T, res_var)
r2 = 1 - (ss_res/ss_tot)
print('The R^2 value is:', r2[0,0])

# PLOTS
t = np.arange(-5, 5, 0.05)
pred_line = theta[0,0] + theta[1,0]*t + theta[2,0]*(t**2) + theta[3,0]*(t**3)

plt.scatter(x_data, y_data, s=20)
plt.scatter(t, pred_line, c = 'g', s = 5)
plt.show()
