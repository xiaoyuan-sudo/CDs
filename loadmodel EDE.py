from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd


load_model = load_model(r'./savedModel./CDsmodelEDE.h5')
df = pd.read_excel(r'./data/data.xlsx',1,header = None)
X = df.values
x_test = X[:,0:-2]
y_test = X[:,-2]
x_test = x_test.reshape((-1,40,1))

load_model.evaluate(x_test,y_test)
