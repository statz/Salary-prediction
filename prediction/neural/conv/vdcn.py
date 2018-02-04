import pandas as pd
import os
import h5py
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.models import Model
from keras.layers.convolutional import Conv1D
from keras.layers.embeddings import Embedding
from keras.layers import Input, Dense, Dropout, Lambda
from keras.layers.pooling import MaxPooling1D
from keras.optimizers import SGD


class ConvBlockLayer(object):
    def __init__(self, input_shape, num_filters):
        self.model = Sequential()
        # first conv layer
        self.model.add(Conv1D(filters=num_filters, kernel_size=3, strides=1, padding="same", input_shape=input_shape))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

        # second conv layer
        self.model.add(Conv1D(filters=num_filters, kernel_size=3, strides=1, padding="same"))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))

    def __call__(self, inputs):
        return self.model(inputs)


def get_conv_shape(conv):
    return conv.get_shape().as_list()[1:]



def build_model(num_filters, sequence_max_length=150, num_quantized_chars=71, embedding_size=10, learning_rate=0.001,
                top_k=3, model_path=None):
    inputs = Input(shape=(sequence_max_length,), dtype='int32', name='inputs')
    inputs1 = Input(shape=(3,), name='inputs1')

    embedded_sent = Embedding(num_quantized_chars, embedding_size, input_length=sequence_max_length)(inputs)

    # First conv layer
    conv = Conv1D(filters=64, kernel_size=3, strides=2, padding="same")(embedded_sent)

    # Each ConvBlock with one MaxPooling Layer
    for i in range(len(num_filters)):
        conv = ConvBlockLayer(get_conv_shape(conv), num_filters[i])(conv)
        conv = MaxPooling1D(pool_size=3, strides=2, padding="same")(conv)

    # k-max pooling (Finds values and indices of the k largest entries for the last dimension)
    def _top_k(x):
        x = tf.transpose(x, [0, 2, 1])
        k_max = tf.nn.top_k(x, k=top_k)
        return tf.reshape(k_max[0], (-1, num_filters[-1] * top_k))

    k_max = Lambda(_top_k, output_shape=(num_filters[-1] * top_k,))(conv)

    # 3 fully-connected layer with dropout regularization
    k_max = keras.layers.concatenate([k_max, inputs1])
    fc1 = Dropout(0.5)(Dense(512, activation='relu', kernel_initializer='he_normal')(k_max))
    fc2 = Dropout(0.5)(Dense(512, activation='relu', kernel_initializer='he_normal')(fc1))
    fc3 = Dense(1, activation='relu')(fc2)

    # define optimizer
    sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=False)
    model = Model(inputs=[inputs, inputs1], outputs=fc3)
    model.compile(optimizer="adam", loss='mse', metrics=['mean_absolute_percentage_error'])

    if model_path is not None:
        model.load_weights(model_path)

    return model


data_path = os.getcwd().split("\\prediction")[0] + "\\test_train\\"
N = 800000
train_ind = pd.read_csv(os.getcwd() + "\\data\\" + 'down_ind_train_x.csv')[:N].as_matrix()
h5f = h5py.File(os.getcwd() + "\\data\\" + "down_vectors_repr.h5", 'r')
embedding_matrix = h5f['dataset_1'][:]
h5f.close()
print(1)
train_y = pd.read_csv(data_path + "down_y_train.csv")["down"].as_matrix()[:N]
#min_y = train_y.min()
#max_y = train_y.max()
#train_y = (train_y) / (max_y - min_y)
train_y = (train_y) / 1000
num_filters = [64, 128, 256, 512]
data = pd.read_csv(data_path + "down_x_train.csv")[["Exp", "EmploymentType", "WorkHours"]][:N].as_matrix()
print(2)
model = build_model(num_filters=num_filters, num_quantized_chars=embedding_matrix.shape[0])
model.fit([train_ind, data], train_y, batch_size=500, epochs=1000, validation_split=0.01)