import h5py
from keras.layers import Dropout
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Input, concatenate
from keras.models import Model
from keras.optimizers import Adam

h5f = h5py.File("data\\train_clusters.h5", 'r')
train_clusters = h5f['dataset_1'][:]
h5f.close()

h5f = h5py.File("data\\test_clusters.h5", 'r')
test_clusters = h5f['dataset_1'][:]
h5f.close()

for i in range(20,21):
    print(i)
    train_indexes = [index for index, value in enumerate(train_clusters) if value == i]
    test_indexes = [index for index, value in enumerate(test_clusters) if value == i]

    h5f = h5py.File("data\\train_x2.h5", 'r')
    train_text = h5f['dataset_1'][train_indexes]
    h5f.close()

    h5f = h5py.File("data\\train_x2.h5", 'r')
    test_text = h5f['dataset_1'][test_indexes]
    h5f.close()

    h5f = h5py.File("data\\train_x5.h5", 'r')
    train_inf = h5f['dataset_1'][train_indexes]
    h5f.close()

    h5f = h5py.File("data\\train_x5.h5", 'r')
    test_inf = h5f['dataset_1'][test_indexes]
    h5f.close()

    h5f = h5py.File("data\\train_y.h5", 'r')
    train_y = h5f['dataset_1'][train_indexes]
    h5f.close()

    h5f = h5py.File("data\\train_y.h5", 'r')
    test_y = h5f['dataset_1'][test_indexes]
    h5f.close()

    input1 = Input(shape=(train_text.shape[1],))
    input2 = Input(shape=(train_inf.shape[1],))
    x1 = Dense(1000, activation='tanh')(input1)
    x1 = Dropout(0.45)(x1)
    x1 = Dense(500, activation='tanh')(x1)
    x1 = Dropout(0.35)(x1)
    x = concatenate([x1, input2])
    x = Dense(200, activation='tanh')(x)
    x = Dropout(0.25)(x)
    x = Dense(50, activation='tanh')(x)
    x = Dropout(0.1)(x)
    preds = Dense(1, activation='linear')(x)

    model = Model([input1, input2], preds)
    opt = Adam()

    model.compile(loss='mean_absolute_percentage_error',
                  optimizer=opt,
                  metrics=['mean_absolute_percentage_error'])
    es = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')

    model.fit([train_text, train_inf],
              train_y, batch_size=int(len(train_indexes) / 50), epochs=200, callbacks=[es], shuffle='random',
              validation_split=0.1)
