''' DECISION TREE ALGORTIHM FOR CLASSIFICATION
It incorportaes a correlation analysis to exlcude redundant variables.
Also, it deals with categorical string variables and transforms them into numerical ones 
This means, no preparation in Excel is necessary. '''


''' Libraries '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn import tree
from sklearn.model_selection import train_test_split


''' Preparing the data '''

# Import the data
data = pd.read_csv("/Users/diego/Python/Machine Learning/bdiag_full.csv", delimiter = ",")

# Extract the variable to explain (categorical) and express it as 0 or 1
targets = np.asarray(data.iloc[:, 1])
encoder = preprocessing.LabelEncoder()
encoder.fit(targets)
# Set the proper dimensions:
targets = encoder.transform(targets)[:, np.newaxis]

# Features: select all columns except 'id' (irrelevant) and 
# 'diagnosis' (variable to explain)
data = data.loc[:, ~data.columns.isin(['id', 'diagnosis'])]

# Create the correlation matrix and remove columns with correlation >= 0.7
correlation_matrix = data.corr().round(2)
upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape),k=1).astype(np.bool))
to_drop = [column for column in upper_tri.columns if any(abs(upper_tri[column]) >= 0.7)]

# Define the reduced dataset
data_red = data.drop(columns=[col for col in data if col in to_drop])

# Correlation matrix of the reduced dataset to double-check:
correlation_matrix = data_red.corr().round(2)
sns.heatmap(data = correlation_matrix, annot=True)
plt.clf() # Comment this to display the heatmap

# Set the features' dimensions and make them floating point:
features = np.asarray(data_red).astype(float)

# List of feature and class names:
feature_names = list(np.asarray(data_red.columns))
class_names = ['B', 'M']


''' Decision tree '''

def build_tree(features, targets, feature_names, class_names):
    # Split the dataset into training and test features:
    train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.9, random_state=100)
    # Create the decision tree
    decision_tree = tree.DecisionTreeClassifier(random_state=456)
    decision_tree = decision_tree.fit(train_features, train_targets)

    # Creates the tree diagram:
    plt.subplots(figsize=(17, 12))
    tree.plot_tree(decision_tree, feature_names=feature_names, filled=True, rounded=False, class_names=class_names)
    plt.savefig("decision_tree.png")

    # Calculates the accuracy:
    train_error = round(decision_tree.score(train_features, train_targets), 2)
    test_error = round(decision_tree.score(test_features, test_targets), 4)
    print("Training Set Mean Accuracy = " + str(train_error))
    print("Test Set Mean Accuracy = " + str(test_error))
    
    # Calculates the accuracy by hand to double-check
    pred = decision_tree.predict(test_features)
    pred = pred[:, np.newaxis]
    print('Accuracy:', round(100*sum(pred == test_targets)[0]/len(pred), 2), "%")

print(build_tree(features, targets, feature_names, class_names))
