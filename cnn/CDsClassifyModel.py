import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class CDsClassifyModel(keras.Model):
    def __init__(self):
        super.__init__(self)
        self.con1 = layers.Dense()
        self.com2 = layers.Dense()


    def call(self, x):
        y1 = self.con1(x)
        y2 = self.com2(y2)


        return output