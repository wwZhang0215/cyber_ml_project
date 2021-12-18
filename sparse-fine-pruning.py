import random
import sys

import keras
import tensorflow as tf
import h5py
import numpy as np


def sparse_fine_pruning(model_path, data_path, X, epochs):
    # init model
    bd_model = keras.models.load_model(model_path)
    prune_model = keras.models.clone_model(bd_model)
    prune_model.set_weights(bd_model.get_weights())

    # print(prune_model.summary())

    # get clean valid data
    eval_data = h5py.File(data_path, 'r')
    x_data = np.array(eval_data['data'])
    y_data = np.array(eval_data['label'])
    x_data = x_data.transpose((0, 2, 3, 1))
    # x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)

    # randomly sparse X% of channels
    sparse_channels = random.sample(range(0, 60), int(60 * X))
    weights = prune_model.get_weights()
    for channel in sparse_channels:
        weights[4][..., channel] = 0
        weights[5][channel] = 0
    prune_model.set_weights(weights)

    # train for one epoch
    prune_model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        optimizer='adam',
        metrics=['accuracy'])

    prune_model.fit(x_data, y_data, epochs=1)

    # sparse channels to same sparsity and repeat training
    for i in range(epochs):
        print(i)
        extractor = keras.Model(inputs=prune_model.input, outputs=prune_model.get_layer('pool_3').output)
        features = extractor(x_data)

        feature_array = features.numpy()

        sum_by_channel = np.sum(feature_array, axis=(0, 1, 2))
        sorted_channel = np.argsort(sum_by_channel)

        weights = prune_model.get_weights()
        for c in range(int(60 * X)):
            channel = sorted_channel[c]
            weights[4][..., channel] = 0
            weights[5][channel] = 0
        prune_model.set_weights(weights)
        prune_model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            optimizer='adam',
            metrics=['accuracy'])

        prune_model.fit(x_data, y_data, epochs=1)

        if i % 10 == 0:
            file_name = "repair_" + model_path.split('/')[1].split('.')[0] + "_" + str(X) + "_" + str(i) + ".h5"
            prune_model.save(file_name)


if __name__ == '__main__':
    model_path = sys.argv[1]
    data_path = sys.argv[2]
    # X = [0.1, 0.2, 0.3, 0.4, 0.5]
    # epochs = 71
    # for k in X:
    #     print(k)
    #     sparse_fine_pruning(model_path, data_path, 0.4, epochs)
    sparse_fine_pruning(model_path, data_path, 0.3, 41)
