import numpy as np
import tensorflow as tf

class Reader:
    def __init__(self):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0
        self.x_train = self.x_train.reshape(self.x_train.shape + (1,))
        self.x_test = self.x_test.reshape(self.x_test.shape + (1,))

        self.train_size = self.x_train.shape[0]

    def get_train_batch(self, batch_size):
        index = np.random.randint(0, self.x_train.shape[0], batch_size)
        return self.x_train[index], self.y_train[index]

    def get_test_batch(self, batch_size):
        index = np.random.randint(0, self.x_test.shape[0], batch_size)
        return self.x_test[index], self.y_test[index]
    
    def get_train_data(self, index):
        return self.x_train[index], self.x_train[index]

    def get_test_data(self, index):
        return self.x_test[index], self.y_test[index]
