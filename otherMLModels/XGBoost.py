import numpy as np
import pandas as pd
from xgboost import XGBClassifier

seed = 7
np.random.seed(seed)
df = pd.read_excel(r'..\data\data.xlsx',header = None)
X = df.values
x_train = X[0:153,0:-7]
y_train = X[0:153,-7]
x_val = X[153:170,0:-7]
y_val = X[153:170,-7]

def accuracy(y_pred,y_true):
    N = y_true.shape[0]
    count = 0
    for i in range(N):
        count = count + np.array_equal(y_pred[i],y_true[i])
    return count/N
xgb = XGBClassifier(n_estimators=100,max_depth = 2)
xgb.fit(x_train, y_train)  
y_pred = xgb.predict(x_val)  
acc = accuracy(y_pred,y_val)
print("val accuracy:",acc)