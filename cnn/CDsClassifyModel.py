import tensorflow as tf
from tensorflow.keras import Model, layers

class CDsClassifyModel(Model):
    def __init__(self):
        super(CDsClassifyModel, self).__init__()
        self.con1 = layers.Conv1D(32, 3, activation='relu', input_shape=(40, 1))
        self.con2 = layers.Conv1D(64, 3, activation='relu')
        self.fla = layers.Flatten()
        self.den1 = layers.Dense(64,activation = 'relu')
        self.den2 = layers.Dense(32,activation = 'relu')
        self.drop = layers.Dropout(0.1)
        self.den3 = layers.Dense(2, activation='softmax')
    
    def call(self, inputs):
        out = self.con1(inputs)
        out = self.con2(out)
        out = self.fla(out)
        out = self.den1(out)
        out = self.den2(out)
        out = self.drop(out)
        y_pred = self.den3(out)
        return y_pred
