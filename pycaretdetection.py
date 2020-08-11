import numpy as np 
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

!pip install pycaret

df=pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
len(df[df['Class']==0])
len(df[df['Class']==1])

from pycaret.classification import *
clf1 = setup(data = df, target = 'Class')

compare_models()

xgboost = create_model('xgboost')

xgboost

model=tune_model('xgboost')