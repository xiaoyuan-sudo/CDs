import numpy as np
import pandas as pd
import tensorflow as tf
from cnn.CDsClassifyModel import CDsClassifyModel


def get_dataset(filename=None):
    if filename == None:
        raise NameError("Please check path of the file!")

    df = pd.read_excel(filename,1,header = None)
    X = df.values
    X_train = X[0:155,:]
    x_train = X_train[:,0:-2]
    y_train = X_train[:,-2]

    X_test = X[155:169, :]
    x_test = X_test[:,0:-2]
    y_test = X_test[:,-2]
    c = 40
    x_train = x_train.reshape((-1,c,1))
    x_test = x_test.reshape((-1,c,1))

    datatrain = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    datatrain = datatrain.shuffle(5).batch(8)
    datatest = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    datatest = datatrain.batch(8)

    return (datatrain, datatest)

def main():
    dataset_train, dataset_test = get_dataset('./data/data.xlsx')

    model = CDsClassifyModel()

    model.compile(optimizer="adam",
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    model.fit(dataset_train, epochs=200)

    model.evaluate(dataset_test, verbose=2)

if __name__ == "__main__":
    main()