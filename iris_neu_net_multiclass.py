# IMPLEMENT A NEURAL NETWORK AS PART OF A MULTI CLASSIFICATION ALGORITHM (3 CATEGORIES)

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils

# Import data and add the 'test' variable:
df = pd.read_csv("/Users/diego/Downloads/iris.csv", delimiter = ";")
test = df['Sepal.Length'] > 5
df['test'] = (df['Sepal.Length'] < 5)

# Extract the target variable ('Species') and express its values as vectors (1,0,0), (0,1,0) and (0,0,1):
targets = np.asarray(df.iloc[:, -2])
encoder = LabelEncoder()
encoder.fit(targets)
encoded_targets = encoder.transform(targets)

targets = np_utils.to_categorical(encoded_targets)

# Select the explicative variables
data1 = df.loc[:, ~df.columns.isin(['test', 'Species'])]

# Create the correlation matrix:
correlation_matrix = data1.corr().round(2)
sns.heatmap(data = correlation_matrix, annot=True)
plt.clf() # Comment this to display the heatmap

# Remove the variables with correlaction >= 0.7
upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape),k=1).astype(np.bool))
to_drop = [column for column in upper_tri.columns if any(abs(upper_tri[column]) >= 0.7)]

# Dataframe without redundant variables:
data_red = data1.drop(columns=[col for col in data1 if col in to_drop])

# New correlation matrix with independent variables:
correlation_matrix = data_red.corr().round(2)
sns.heatmap(data = correlation_matrix, annot=True)
plt.clf() # Comment this to display the heatmap

# Define the model variables:
variables = np.asarray(data_red).astype(float)

# Import some libraries for the neural network:
from keras.models import Sequential
from keras.layers import Dense

# Define the network's architecture:
model = Sequential()
model.add(Dense(12, input_dim = 2, activation='relu'))
model.add(Dense(3, activation ='softmax'))

# Train the network and define its history:
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
history = model.fit(variables, targets, epochs=300, batch_size = 10, verbose=0)

# Model's accuracy:
_, accuracy = model.evaluate(variables, targets)
print('Accuracy:', round(accuracy*100, 2), '%')

# Plot the accuracy and the cost function as a function of the epoch:
plt.plot(history.history['accuracy'])
plt.title('Accuracy')
plt.ylabel("Network's accuracy")
plt.xlabel("Epoch")
plt.show()

plt.plot(history.history['loss'])
plt.title('Cost function')
plt.ylabel('Cost')
plt.xlabel('Epoch')
plt.show()

# Plot the two independent variables and group the data according to the 'Species' variable:
from sklearn import preprocessing

y = np.asarray(df.iloc[:, -2])
encoder = preprocessing.LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)[:, np.newaxis]

seto = np.argwhere(y == 0)
vers = np.argwhere(y == 1)
virg = np.argwhere(y == 2)
    
plt.scatter(variables[seto, 0], variables[seto, 1], c ='r', label = 'Setosa')
plt.scatter(variables[vers, 0], variables[vers, 1], c ='b', label = 'Versicolor')
plt.scatter(variables[virg, 0], variables[virg, 1], c ='g', label = 'Virginica')
plt.xlabel('Sepal.Length')
plt.ylabel('Sepal.Width')
plt.legend()
plt.show()
