# IMPLEMENTS A CLASSIFICATION ALGORITHM USING A DECISION TREE

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn import tree
from sklearn.model_selection import train_test_split

# Import and add the 'Test' variable:
df = pd.read_csv("/Users/diego/Downloads/iris.csv", delimiter = ";")
test = df['Sepal.Length'] > 5
df['test'] = (df['Sepal.Length'] < 5)

# Extract the target variable ('Test') and express it as 0 and 1:
targets = np.asarray(df.iloc[:, -1])
encoder = preprocessing.LabelEncoder()
encoder.fit(targets)
targets = encoder.transform(targets)[:, np.newaxis]

# Select the variables:
# NOTE: 'Sepal.Length' is not included since this one and 'Test' are basically the same.
data1 = df.loc[:, ~df.columns.isin(['test', 'Species', 'Sepal.Length'])]

# Create the correlation matrix:
correlation_matrix = data1.corr().round(2)
sns.heatmap(data = correlation_matrix, annot=True)
plt.clf() # Comment this to display the heatmap

# Remove variables with correaltion >= 0.7
upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape),k=1).astype(np.bool))
to_drop = [column for column in upper_tri.columns if any(abs(upper_tri[column]) >= 0.7)]

# Dataframe without redundant variables:
data_red = data1.drop(columns=[col for col in data1 if col in to_drop])

# New correlation matrix:
correlation_matrix = data_red.corr().round(2)
sns.heatmap(data = correlation_matrix, annot=True)
plt.clf() # Comment this to display the heatmap

# Define the surviving variables as features:
features = np.asarray(data_red).astype(float)
feature_names = list(np.asarray(data_red.columns))
class_names = ['F', 'T']

# Decsion tree:
def build_tree(features, targets, feature_names, class_names):
    # Split the data into train and test:
    train_features, test_features, train_targets, test_targets = train_test_split(features, targets, test_size=0.25, random_state=123)
    # Create the decision tree:
    decision_tree = tree.DecisionTreeClassifier(random_state=456)
    decision_tree = decision_tree.fit(train_features, train_targets)

    # Generate the tree diagram:
    plt.subplots(figsize=(17, 12))
    tree.plot_tree(decision_tree, feature_names=feature_names, filled=True, rounded=False, class_names=class_names)
    plt.savefig("decision_tree.png")

    # Compute the accuracy:    
    pred = decision_tree.predict(test_features)
    pred = pred[:, np.newaxis]
    print('Accuracy:', round(100*sum(pred == test_targets)[0]/len(pred), 2), "%")

print(build_tree(features, targets, feature_names, class_names))
