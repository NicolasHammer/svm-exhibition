# Import relevant packages
import numpy as np
import pandas as pd

import from_scratch.split_data as split_data
import from_scratch.svm as svm

# Load data
original_data = pd.read_csv("data/breast_cancer.csv", sep=',')
original_data.head()

# Clean data
diagnosis_map = {'M': 1, 'B': - 1}
original_data["diagnosis"] = original_data["diagnosis"].map(diagnosis_map)
original_data.drop(original_data.columns[[-1, 0]], axis=1,  inplace=True)
original_data.head()

# Split into train and test and normalize
Y = original_data.loc[:, "diagnosis"]
X = original_data.iloc[:, 1:]

X_normalized = (X - X.min())/(X.max() - X.min())
X_normalized.head()

# Train/test split
X_normalized.insert(loc=len(X_normalized.columns),
                    column="intercept", value=1)  # add bias to features

train_features, train_targets, test_features, test_targets = split_data.train_test_split(
    X_normalized.values.T, Y.values.reshape((1, Y.shape[0])))

# Train model
max_epochs = 500
svm_model = svm.SVM(train_features.shape[0])
svm_model.fit(train_features, train_targets)