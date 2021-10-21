from pprint import pprint
import pandas as pd

# # cleaning negative values
# # TODO: why are there negative values ?
# def clean_negative_values(df):
#   s=(df<0).sum()
#   cols=s[s>0].index

#   for col in cols:
#     ii=df[df[col]<0].index
#     df.loc[ii, col]=0

# def print_negative_values(df):
#   s=(df<0).sum()
#   cols=s[s>0].index

#   pprint([(col, df[df[col]<0].index, df[df[col]<0][col].values) for col in cols])


def read_clean_classify_data(miRna_path, predictions_path):
  data=pd.read_csv(miRna_path)
  df=data.drop(data.columns[0], axis=1)
  df=df.T

  rna=data['miRNA'].values
  df.columns=rna

  s=(df<0).sum()
  cols=s[s>0].index

  for col in cols:
    ii=df[df[col]<0].index
    df.loc[ii, col]=0

  df_predictions=pd.read_csv(predictions_path)
  
  s_0=set(df_predictions[df_predictions['Stage']=='*']['Histology'].unique())
  s_1=set(df_predictions[df_predictions['Stage']!='*']['Histology'].unique())

  s_0=s_0-set(['endometrioid adenocarcinoma'])
  s_1=s_1-set(['serous borderline'])

  df_predictions['cancer']=df_predictions['Histology'].isin(s_0).apply(int)

  dx=pd.merge(df, df_predictions.set_index('ID')['cancer'], left_index=True, right_index=True, how='left')

  return dx


