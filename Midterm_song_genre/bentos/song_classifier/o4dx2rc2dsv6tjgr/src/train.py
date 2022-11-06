#!/usr/bin/env python
# coding: utf-8

# # Predict song genre based on audio features (Spotify dataset)

# In[498]:


from pathlib import Path
import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mutual_info_score, mean_squared_error, accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import xgboost as xgb 
import bentoml


# # Load (small) dataset with selected columns

# In[499]:


# Load and display head
df = pd.read_csv("./data/spotify_songs_small.csv")
print(f"Columns:\n{df.columns.values}")
df.head()


# In[500]:


# Visualize basic info
df.describe()


# In[501]:


# Check for missing values
df.isna().sum()


# In[502]:


# Get numeric and categorical columns
numeric_columns = df.dtypes[(df.dtypes == 'float64') | (df.dtypes == 'int64')].index.values
categorical_columns = df.dtypes[df.dtypes == 'object'].index.values
print(f"Numeric columns: {numeric_columns}")
print(f"Categorical columns: {categorical_columns}")


# # Train, validation, test split

# In[503]:


# Extract target column
target_column = 'category'
#y = df.pop(target_column)
print(df[target_column].value_counts())
remapping_to_numbers = {"rock" : 0, 
                        "indie" : 1, 
                        "pop": 2, 
                        "metal" : 3, 
                        "hiphop": 4,
                        "alternative": 5,
                        "blues": 6}

remapping_to_numbers = {"rock" : 1, 
                        "indie" : 0, 
                        "pop": 0, 
                        "metal" : 1, 
                        "hiphop": 0,
                        "alternative": 1,
                        "blues": 0}

df[target_column]=df[target_column].replace(remapping_to_numbers)
df[target_column].value_counts()


# In[504]:


# Splitting the dataset 60%-20%-20% (train-val-test)
#df_train_val, df_test, y_train_val, y_test = train_test_split(df, y, test_size = 0.2, random_state=1)
#df_train, df_val, y_train, y_val = train_test_split(df_train_val, y_train_val, test_size = 0.25, random_state=1)

df_train_val, df_test = train_test_split(df, test_size = 0.2, random_state=1)
df_train, df_val = train_test_split(df_train_val, test_size = 0.25, random_state=1)


# # Initial feature selection (based on feature distribution)

# In[505]:


# Select features
selected_columns = ['danceability',
                    'energy',
                    target_column]

df_train = df_train[selected_columns]
df_val = df_val[selected_columns]
df_test = df_test[selected_columns]


# # Extract target value and features

# In[506]:


y_train = df_train.pop(target_column).values
X_train = df_train.values


# In[507]:


y_val = df_val.pop(target_column).values
X_val = df_val.values


# In[508]:


y_test = df_test.pop(target_column).values
X_test = df_test.values


# # Train final model

# In[509]:


# Select XGB Classifier model based on the previous experiments
xgb_params = {
                'eta': 0.1, 
                'max_depth': 3,
                'min_child_weight': 10,
                'objective': 'reg:logistic',
                'eval_metric': 'auc',

                'nthread': 8,
                'seed': 1,
                'verbosity': 1,
            }


# Retrain on training and validation sets
X_train_full = np.vstack((X_train, X_val))
y_train_full = np.concatenate((y_train, y_val))
dtrain_full = xgb.DMatrix(X_train_full, label=y_train_full, feature_names=df_train.columns.values)
watchlist = [(dtrain_full,'train'), (dtest,'eval')]
xgb_classifier = xgb.train(xgb_params, dtrain_full, evals=watchlist, num_boost_round=50)


# In[510]:


# Evaluation on the training set
y_pred_train_full = xgb_classifier.predict(dtrain_full)
fpr_train_full, tpr_train_full, thresholds_test = roc_curve(y_train_full, y_pred_train_full)
print(f"Training roc_auc_score: {roc_auc_score(y_train_full, y_pred_train_full)}")

# Evaluation on  the test set
y_pred_test = xgb_classifier.predict(dtest)
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred_test)
print(f"Test roc_auc_score: {roc_auc_score(y_test, y_pred_test)}")

# Plotting ROCs
plt.plot(fpr_train_full, tpr_train_full, color='blue', label="final model on full training set")
plt.plot(fpr_test, tpr_test, color='green', label="final model on test set")

plt.plot([0, 1], [0, 1], color='black', label="random model")
plt.plot([0,0,1], [0,1,1], color='red', label="ideal model")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC for the final selected model")
plt.legend()
plt.show()


# # Export model using BentoML

# In[511]:


# Export model to BentoML
model = xgb_classifier
bentoml.xgboost.save_model('rock_alt_metal_song_genre_model', model, signatures={"predict": {"batchable": True}})


# In[525]:


# Check the export by importing the model
booster = bentoml.xgboost.load_model("rock_alt_metal_song_genre_model:latest")

print(f"Predicted value: {booster.predict(xgb.DMatrix([df_test.iloc[10].values], feature_names=df_train.columns.values))}, expected value: {y_test[10]}")
print(f"Predicted value: {booster.predict(xgb.DMatrix([df_test.iloc[2].values], feature_names=df_train.columns.values))}, expected value: {y_test[2]}")

