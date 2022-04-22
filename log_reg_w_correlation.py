
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())
import seaborn as sns

# Import data
data = pd.read_csv("/Users/diego/Python/Machine Learning/bdiag.csv", delimiter = ",")   

# Correlation matrix
correlation_matrix = data.corr().round(2)
sns.heatmap(data = correlation_matrix, annot=True)
plt.clf()   # Comment this to display the heatmap

# Extract columns with correlation >= 0.7 and put then into the 'to_drop' list
upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape),k=1).astype(np.bool))
to_drop = [column for column in upper_tri.columns if any(abs(upper_tri[column]) >= 0.7)]

# Remove high correlation columns from the data.
# NOTE: the last column (variable to explain) is removed since
# it is obviously correlated to the others
data_red = data.drop(columns=[col for col in data if col in to_drop])

# Correlation matrix of the reduced dataset to double-check:
correlation_matrix = data_red.corr().round(2)
sns.heatmap(data = correlation_matrix, annot=True)
plt.clf() # Comment this to display the heatmap

# Define matrix X and vector y
X_temp = np.asarray(data_red)
y = np.asarray(data.iloc[:, -1])
X_temp = X_temp.astype(float)
y = y[:, np.newaxis] # Set the proper dimensions
m, n = X_temp.shape[0], X_temp.shape[1]

#Normalisation function
def norm(X):
    mean = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mean)/sigma
    X_final = np.hstack((np.ones((m,1)), X_norm))
    return X_final

X_norm = norm(X_temp)

#Sigmoid function
def sigmoid(z):
    return 1./(1+np.exp(-z))

#Cost function and gradient 
theta_0 = np.zeros((n+1, 1)) # Initial theta

def cost(theta, X, y):
    h = sigmoid(X@theta)
 #   id = np.ones((m,1))
 #   J = (-1./m)*(y.T@np.log(h) + (id - y).T@np.log(id-h))
    grad = (1./m)*(X.T@(h-y))
    return grad

#Gradient descent (main function)
def grad_des(theta, X, y, alpha, num_iter):
    for i in range(num_iter):
        grad = cost(theta, X, y) 
        theta = theta - (alpha*grad)
    return theta

theta = grad_des(theta_0, X_norm, y, 1, 100) # Optimsed value of theta
print(theta)

#Calculate accuracy
p = np.zeros((m,1))
def predictions(theta, X):
    for i in range(m):
        pred = sigmoid(X@theta)
        if pred[i] >= 0.5: p[i] = 1
        else: p[i]= 0 
    return p

p = predictions(theta, X_norm)
print('Accuracy:', round(100*sum(p == y)[0]/m, 2), "%")

pos = np.argwhere(y == 1)
neg = np.argwhere(y == 0)


x = np.arange(-2, 2, 0.1)
boundary = (-theta[0,0] - x*theta[1,0])/theta[2,0]

plt.scatter(X_norm[pos,1], X_norm[pos,2], c ='r')
plt.scatter(X_norm[neg,1], X_norm[neg,2], c ='b')
plt.plot(x, boundary, c = 'black')
#plt.plot(x, boundary, c = 'black')
plt.show()
