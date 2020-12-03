from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection  import cross_val_score
from sklearn.ensemble import RandomForestClassifier

seed = 7
np.random.seed(seed)
df = pd.read_excel(r'..\data\data.xlsx',header = None)
X = df.values
x_train = X[0:153,0:-7]
y_train = X[0:153,-7]
x_val = X[153:170,0:-7]
y_val = X[153:170,-7]

rf = RandomForestClassifier()

rf.fit(x_train,y_train)
y_pred = rf.predict(x_val)
def accuracy(y_pred,y_true):
    N = y_true.shape[0]
    count = 0
    for i in range(N):
        count = count + np.array_equal(y_pred[i],y_true[i])
    return count/N
acc = accuracy(y_pred,y_val)
print("val accuracy:",acc)