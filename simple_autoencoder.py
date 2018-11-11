from keras.models import Model
from keras.layers import Input, Dense
import numpy as np


def simple_auto_encode(input_matrix, compression_factor=1.1, encode_activation='tanh', decode_activation='linear',
                       optimizer='adadelta', loss='mean_squared_error'):
    _, input_dim = input_matrix.shape
    encoding_dim = round(input_dim / compression_factor)

    input_layer = Input(shape=(input_dim,))
    encode_layer = Dense(encoding_dim, activation=encode_activation)(input_layer)
    decode_layer = Dense(input_dim, activation=decode_activation)(encode_layer)

    auto_encoder = Model(input_layer, decode_layer)
    auto_encoder.compile(optimizer=optimizer, loss=loss)
    auto_encoder.fit(input_matrix, input_matrix, epochs=10, batch_size=100)

    return auto_encoder.predict(input_matrix)


if __name__== "__main__":
    pca_matrix = np.fromfile('data/matrix_pca').reshape(1048575, 10)
    result = simple_auto_encode(pca_matrix)
    np.savetxt('data/reproduce.txt', result, fmt='%s')
    read_result = np.loadtxt('data/reproduce.txt', dtype='float32')
