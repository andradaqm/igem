from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

class RocResults:
  def __init__(self, fpr, tpr, auc, model):
    self.fpr=fpr
    self.tpr=tpr
    self.auc=auc

    self.model=model
  
class InputModel:
  def __init__(self, model_class, scoring_function_name, params, label_name):
    self.model_class=model_class
    self.scoring_function_name = scoring_function_name
    self.params=params
    self.label_name=label_name

class Runner:
  def __init__(self, df, inputs, **kwargs):
    self.df=df
    self.X=df[df.columns[:-1]].values
    self.y=df['cancer']

    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, **kwargs)

    self.inputs=inputs
  
  def fit_roc(self, model_class, scoring_function_name, **kwargs):
    kk = {x:kwargs[x] for x in kwargs if x!='last_column'}
    model_instance = model_class(**kk)

    # pipe = make_pipeline(StandardScaler(), model_instance)
    pipe = make_pipeline(model_instance)

    # model_instance.fit(X_train, y_train)
    pipe.fit(self.X_train, self.y_train)

    # y_pred=model_instance.__getattribute__(scoring_function_name)(X_test)
    y_pred=pipe.__getattribute__(scoring_function_name)(self.X_test)

    if (('last_column' in kwargs) and (kwargs['last_column'])):
      y_pred=y_pred[:,1]

    fpr, tpr, threshold = roc_curve(self.y_test, y_pred)
    auc_res = auc(fpr, tpr)

    # return RocResults(fpr, tpr, auc_res, model_instance)
    return RocResults(fpr, tpr, auc_res, pipe)

  def get_results(self):
    results={}
    for input_model in self.inputs:
      model = input_model.model_class
      params= input_model.params
      scoring_function_name=input_model.scoring_function_name

      res=self.fit_roc(model, scoring_function_name, **params)
      results[input_model.label_name]=res
    
    return results

