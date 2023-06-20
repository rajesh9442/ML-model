import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Load the training and test datasets
train_df = pd.read_csv("C:/Users/rajes/Downloads/train.csv")
test_df = pd.read_csv("C:/Users/rajes/Downloads/test.csv")

x=train_df.drop(columns='Class')
y=train_df.Class

# Encode the 'Id' and 'EJ' columns in the training dataset
label_encoder = LabelEncoder()
train_df['Id'] = label_encoder.fit_transform(train_df['Id'])
train_df['EJ'] = label_encoder.fit_transform(train_df['EJ'])

test_df['Id'] = label_encoder.fit_transform(test_df['Id'])
test_df['EJ'] = label_encoder.fit_transform(test_df['EJ'])

X_train = train_df.drop('Class', axis=1)
Y_train = train_df['Class']

imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression(solver='saga', max_iter=5000)
model.fit(X_train_scaled, Y_train)

X_test = test_df.copy()
X_test = imputer.transform(X_test)
X_test_scaled = scaler.transform(X_test)

Y_test_pred = model.predict(X_test_scaled)

print(Y_test_pred)
