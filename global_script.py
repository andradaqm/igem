import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib.pyplot as plt

from igem.data import read_clean_classify_data

df=read_clean_classify_data('miRNA_cancer.csv', 'model_predictions.csv')

#---------------
from igem.feature_selection import pearson, lasso, rfe

selected_pearson=pearson(df)
df_pearson = df[selected_pearson]
df_pearson['cancer']=df['cancer']

# # selected_rfe=rfe(df[[*df.columns[:10], 'cancer']])
# selected_rfe=rfe(df)
# df_rfe=df[selected_rfe]
# df_rfe['cancer']=df['cancer']

selected_lasso=lasso(df)
df_lasso=df[selected_lasso]
df_lasso['cancer']=df['cancer']

# # selected_pearson_rfe=rfe( df_pearson[[*df_pearson.columns[:10], 'cancer']]  )
# selected_pearson_rfe=rfe(df_pearson)
# df_pearson_rfe=df_pearson[selected_pearson_rfe]
# df_pearson_rfe['cancer']=df_pearson['cancer']

selected_pearson_lasso=lasso(df_pearson)
df_pearson_lasso=df_pearson[selected_pearson_lasso]
df_pearson_lasso['cancer']=df_pearson['cancer']

#---------------
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from igem.models import Runner, InputModel

kwargs={'test_size':0.33, 'random_state':42}
inputs=[
  InputModel(
    model_class = LogisticRegression,
    params={'max_iter':1000},
    scoring_function_name='decision_function',
    label_name='Logistic'
  ),
  InputModel(
    model_class = SVC,
    params={'kernel': 'linear', 'random_state': 4},
    scoring_function_name='decision_function',
    label_name='SVM'
  ),
  InputModel(
    model_class = MLPClassifier,
    params={'hidden_layer_sizes': (9,9,9), 'max_iter': 10000, 'last_column':True},
    scoring_function_name='predict_proba',
    label_name='MLP'
  ),
  InputModel(
    model_class = HistGradientBoostingClassifier,
    params={'last_column':True},
    scoring_function_name='predict_proba',
    label_name='GB'
  )
]

runner_without=Runner(df, inputs, **kwargs)
res_without=runner_without.get_results()
auc_without={k:v.auc for k, v in res_without.items()}

runner_pearson=Runner(df_pearson, inputs, **kwargs)
res_pearson=runner_pearson.get_results()
auc_pearson={k:v.auc for k, v in res_pearson.items()}

runner_rfe=Runner(df_rfe, inputs, **kwargs)
res_rfe=runner_rfe.get_results()
auc_rfe={k:v.auc for k, v in res_rfe.items()}

runner_lasso=Runner(df_lasso, inputs, **kwargs)
res_lasso=runner_lasso.get_results()
auc_lasso={k:v.auc for k, v in res_lasso.items()}

runner_pearson_rfe=Runner(df_pearson_rfe, inputs, **kwargs)
res_pearson_rfe=runner_pearson_rfe.get_results()
auc_pearson_rfe={k:v.auc for k, v in res_pearson_rfe.items()}

runner_pearson_lasso=Runner(df_pearson_lasso, inputs, **kwargs)
res_pearson_lasso=runner_pearson_lasso.get_results()
auc_pearson_lasso={k:v.auc for k, v in res_pearson_lasso.items()}

selected_features={
  'pearson' : selected_pearson,
  # 'rfe' : selected_rfe,
  'lasso' : selected_lasso,
  # 'pearson_rfe' : selected_pearson_rfe,
  'pearson_lasso' : selected_lasso,
}

import json
with open('select_columns.json', 'w') as f:
  json.dump(selected_features, f, indent=2)

df_auc=pd.DataFrame({
  'without' : auc_without,
  'pearson' : auc_pearson,
  'rfe' : auc_rfe,
  'lasso' : auc_lasso,
  'pearson_rfe' : auc_pearson_rfe,
  'pearson_lasso' : auc_pearson_lasso,
})

df_auc.to_csv('auc.csv', index=True)

#---------------------------
all_selected=set([])
for key in selected_features:
  if key!='pearson':
    all_selected=all_selected.union(set(selected_features[key]))

res=[]
for key in selected_features:
  res_item=[]
  for feature in all_selected:
    if (feature in selected_features[key]):
      res_item.append('X')
    else:
      res_item.append('')
  res.append(res_item)  

df_all_selected=pd.DataFrame(res)
df_all_selected.columns=all_selected
df_all_selected.index = [key for key in selected_features]

#----------
df_elias=pd.read_csv('features_elias.csv')

df_tmp=df_elias.melt()
df_tmp=df_tmp[df_tmp['value'].notnull()]



for column in df_elias.columns:
  s_col = df_elias[column]

