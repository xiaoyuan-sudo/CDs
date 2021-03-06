#SVM for CDs prediction
import pandas as pd
import numpy as np
from sklearn.svm import SVC

df = pd.read_excel(r'..\data\data.xlsx',header = None)
X = df.values
x_train = X[0:153,0:-7]
y_train = X[0:153,-7]
x_val = X[153:170,0:-7]
y_val = X[153:170,-7]
model = SVC(C = 100, kernel='rbf', probability=True,decision_function_shape='ovo')
model.fit(x_train,y_train)
y_pred = model.predict(x_val)
def accuracy(y_pred,y_true):
    N = y_true.shape[0]
    count = 0
    for i in range(N):
        count = count + np.array_equal(y_pred[i],y_true[i])
    return count/N
acc = accuracy(y_pred,y_val)
print("val accuracy:",acc)