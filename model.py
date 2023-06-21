import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Load the training and test datasets
train_df = pd.read_csv("/content/train.csv")
test_df = pd.read_csv("/content/test.csv")

# Perform one hot encoding on training dataset
train_df_new = pd.get_dummies(train_df, columns=['EJ'],  dtype=float).dropna()

# Split the training dataset into features (X) and target labels (Y)
X_train = train_df_new.drop(['Id','Class'], axis=1)
Y_train = train_df_new['Class']

#Training the model
model = LogisticRegression(solver='saga', max_iter=5000)
model.fit(X_train, Y_train)

# Perform onehot encoding for test data
test_df.dtypes
X_test = test_df.drop(['Id'],axis=1)
X_test = pd.get_dummies(X_test, columns=['EJ'],  dtype=float, sparse=False)

#Predicting the output
Y_test_pred = model.predict(X_test)

print(Y_test_pred)
