import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import copy
from environmental_raster_glc import PatchExtractor
import tensorflow as tf
from tensorflow.contrib import tpu
from tensorflow.contrib.cluster_resolver import TPUClusterResolver
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from sklearn.preprocessing import StandardScaler
model = Sequential()
model.add(Conv2D(128, kernel_size = 2, input_shape = (77, 8, 8), activation = 'relu'))
#output dimension: 7x7x128xbatch size
model.add(Conv2D(128, kernel_size = 2, activation = 'relu')) 
#output dimension: 6x6x128xbatch size
model.add(MaxPooling2D(pool_size = 2, strides = 1))
#output dimension: 5x5x128xbatch size
model.add(Conv2D(64, kernel_size = 2, activation = 'relu'))
#output dimension: 4x4x64xbatch size
model.add(Flatten())
#output dimension: 1024x1xbatch size
model.add(Dense(units = 2000, activation = 'relu'))
#output dimension: 2000x1xbatch size
model.add(Dense(units = 2000, activation = 'relu'))
#output dimension: 2000x1xbatch size
model.add(Dense(units = 2000, activation = 'relu'))
#output dimension: 2000x1xbatch size
model.add(Dense(units = 2000, activation = 'relu'))
#output dimension: 2000x1xbatch size
model.add(Dense(units = 1348, activation = 'softmax'))
#output dimension: 1348x1xbatch size(probability values for classes)
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
def fit_and_predict(X_train, y_train, Xtest, y_test):
    global model
    
    model.fit(x = X_train, y = y_train, epochs = 5, batch_size = 2000)
    (loss, accuracy) = model.evaluate(x = X_test, y = y_test)
    print('Loss: {} Accuracy: {}'.format(loss, accuracy * 100))
class GeoLifeClefDataset:
    def __init__(self, extractor, dataset, labels):
        self.extractor = extractor
        self.labels = labels
        self.dataset = dataset
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        tensor = self.extractor[self.dataset[idx]]
        return tensor, self.labels[idx]


if __name__ == '__main__':
    patch_extractor = PatchExtractor('../rasters GLC19', size=8, verbose=True)
    patch_extractor.add_all()
    # dataset
    df = pd.read_csv("../PL_trusted.csv",sep=';')
    classes = set(df['glc19SpId'])
    df = pd.concat([df.drop('glc19SpId',axis=1),pd.get_dummies(df['glc19SpId'],dtype=int)], axis=1)
    dataset_list = list(zip(df["Latitude"],df["Longitude"]))
    labels_list = (df.iloc[:, 10:]).values
    train_ds = GeoLifeClefDataset(patch_extractor, dataset_list[:230000], labels_list[:230000])
    test_ds = GeoLifeClefDataset(patch_extractor, dataset_list[230000:], labels_list[230000:])
    datasets = {"train": train_ds, "val": test_ds}
    
    X_train, y_train, X_test, y_test = [], [], [], []
    for i in range(len(train_ds)):
        X_train.append(train_ds[i][0])
        y_train.append(train_ds[i][1])
    for i in range(len(test_ds)):
        X_test.append(test_ds[i][0])
        y_test.append(test_ds[i][1])
    X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
    """
    # TO USE TPU
    inputs = [X_train, y_train, X_test, y_test]
    tpu_computation = tpu.rewrite(fit_and_predict, inputs)
    tpu_grpc_url = TPUClusterResolver(tpu = [os.environ['TPU_NAME']]).get_master()
    with tf.Session(tpu_grpc_url) as sess:
        sess.run(tpu.initialize_system())
        sess.run(tf.global_variables.initialize())
        output = sess.run(tpu_computation)
        print('Output: ', output)
        sess.run(tpu.shutdown_system())
    print('Done')
    """
    fit_and_predict(X_train, y_train, X_test, y_test)
    model.save('CNN_Model.h5')
