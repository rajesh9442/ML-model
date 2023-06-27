import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Load the training and test datasets
train_df = pd.read_csv("C:/Users/rajes/Downloads/train.csv")
test_df = pd.read_csv("C:/Users/rajes/Downloads/test.csv")

encoder = ColumnTransformer([('onehot', OneHotEncoder(), ['EJ'])], remainder='passthrough')

# Split the training dataset into features (X) and target labels (Y)
X_train = encoder.fit_transform(train_df.drop(['Id', 'Class'], axis=1))
Y_train = train_df['Class']

# Handle missing values in training dataset
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)

# Create a decision tree classifier
model = DecisionTreeClassifier()
model.fit(X_train, Y_train)

# Perform one-hot encoding for test data
X_test = encoder.transform(test_df.drop(['Id'], axis=1))

# Handle missing values in the test dataset
X_test = imputer.transform(X_test)

# Convert back to DataFrames for column alignment
X_train = pd.DataFrame(X_train, columns=encoder.get_feature_names_out(train_df.columns.drop(['Id', 'Class'])))
X_test = pd.DataFrame(X_test, columns=encoder.get_feature_names_out(test_df.columns.drop(['Id'])))

# Align columns of test dataset with training dataset
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Predicting the output
Y_test_pred = model.predict(X_test)

# Convert Y_test_pred to a DataFrame
Y_test_pred_df = pd.DataFrame(Y_test_pred, columns=['Class'])

# Create the submission DataFrame
submission_df = pd.DataFrame({
    'Id': test_df['Id'],
    'class_0': Y_test_pred_df['Class'].apply(lambda x: 1 - x),
    'class_1': Y_test_pred_df['Class']
})

# Save the submission DataFrame to a CSV file
submission_df.to_csv('submission.csv', index=False)

# Define the file paths
output_path = 'submission.csv'
download_path = 'C:/Users/rajes/Downloads/submission.csv'







