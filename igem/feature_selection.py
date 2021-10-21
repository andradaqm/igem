import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

##TODO : warning to df : it is implicit that the last column is ignore.
def pearson(df, threshold=.8):
  X_columns=df.columns[:-1]
  corr = df[X_columns].corr()
  columns = np.full((corr.shape[0],), True, dtype=bool)
  for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
      if corr.iloc[i,j] >= threshold:
        if columns[j]:
          columns[j] = False

  selected_columns = X_columns[columns].values
  return sorted(selected_columns)

def lasso(df):
  X=df[df.columns[:-1]]
  y=df['cancer']

  features=df.columns[:-1]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
  pipeline = Pipeline([
    ('scaler',StandardScaler()),
    ('model',Lasso())
  ])

  search = GridSearchCV(pipeline,
    {'model__alpha':np.arange(0.1,10,0.1)},
    cv = 5, #cross validation
    scoring="neg_mean_squared_error",
    verbose=0)

  search.fit(X_train,y_train)

  search.best_params_

  coefficients = search.best_estimator_.named_steps['model'].coef_

  importance = np.abs(coefficients)

  selected_columns=np.array(features)[importance>0]

  return list(selected_columns)

def rfe(df):
  X=df[df.columns[:-1]]
  y=df['cancer']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

  nof_list=np.arange(1,X.shape[1]+1)
  # nof_list=np.arange(1,2+1)
  high_score=0
  #Variable to store the optimum features
  nof=0           
  score_list =[]

  model = LogisticRegression(max_iter=1000)

  for n in nof_list:
    rfe = RFE(model,n_features_to_select=n)
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    X_test_rfe = rfe.transform(X_test)
    model.fit(X_train_rfe,y_train)
    score = model.score(X_test_rfe,y_test)
    score_list.append(score)
    if(score>high_score):
      high_score = score
      nof = nof_list[n-1]
  # print("Optimum number of features: %d" %nof)
  # print("Score with %d features: %f" % (nof, high_score))

  #Initializing RFE model
  # nof=10
  rfe = RFE(model,n_features_to_select= nof)
  #Transforming data using RFE
  X_train_rfe = rfe.fit_transform(X_train,y_train)
  X_test_rfe = rfe.transform(X_test)  
  #Fitting the data to model
  model.fit(X_train_rfe,y_train)

  features=df.columns[:-1]
  # len(features)
  # len(rfe.support_)

  return list(features[rfe.support_])


