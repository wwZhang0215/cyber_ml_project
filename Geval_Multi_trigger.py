import keras
import sys
import h5py
import numpy as np


def data_loader(filepath):
    data = h5py.File(filepath, 'r')
    x_data = np.array(data['data'])
    y_data = np.array(data['label'])
    x_data = x_data.transpose((0, 2, 3, 1))

    return x_data, y_data


def G(bd_model, prune_model, bd_x_test, bd_y_test):
    bd_label_p = np.argmax(bd_model.predict(bd_x_test), axis=1)
    prune_label_p = np.argmax(prune_model.predict(bd_x_test), axis=1)
    attack_label = np.max(bd_y_test) + 1
    G_label = []

    for i in range(len(bd_y_test)):
        label = bd_label_p[i] if bd_label_p[i] == prune_label_p[i] else attack_label
        G_label.append(label)

    return np.array(G_label)


def main(clean_data_filename, poisoned_data_filenames, bd_model_filename, prune_model_filename):
    clean_accuracies = 0.0
    asrs = 0.0
    cl_x_test, cl_y_test = data_loader(clean_data_filename)

    bd_model = keras.models.load_model(bd_model_filename)
    prune_model = keras.models.load_model(prune_model_filename)

    cl_label_p = np.argmax(prune_model.predict(cl_x_test), axis=1)
    clean_accuracy = np.mean(np.equal(cl_label_p, cl_y_test)) * 100

    clean_accuracies += clean_accuracy
    for poisoned_data_filename in poisoned_data_filenames:
        bd_x_test, bd_y_test = data_loader(poisoned_data_filename)

        bd_label_p = G(bd_model, prune_model, bd_x_test, bd_y_test)
        asr = np.mean(np.equal(bd_label_p, bd_y_test)) * 100
        print('Attack Success Rate on ', poisoned_data_filename, ':', asr)
        asrs += asr

    print('Average Attack Success Rate:', asrs / len(poisoned_data_filenames))
    print('Clean Classification accuracy :', clean_accuracy)


if __name__ == '__main__':
    clean_data_filename = "data/clean_test_data.h5"
    poisoned_data_filename = ["data/Multi-trigger/eyebrows_poisoned_data.h5",
                              "data/Multi-trigger/lipstick_poisoned_data.h5",
                              "data/Multi-trigger/sunglasses_poisoned_data.h5"]
    bd_model_filename = "models/multi_trigger_multi_target_bd_net.h5"
    prune_model_filename = "models/repair_multi_trigger_multi_target_bd_net.h5"
    main(clean_data_filename, poisoned_data_filename, bd_model_filename, prune_model_filename)

    # for i in [0.1, 0.2, 0.3, 0.4, 0.5]:
    # for i in [0.1, 0.2, 0.3, 0.4, 0.5]:
    #     for j in [0, 10, 20, 30, 40, 50, 60, 70]:
    #         prune_model_filename = "repair_multi_trigger_multi_target_bd_net_" + str(i) + "_" + str(j) + ".h5"
    #         print(prune_model_filename)
    #         main(clean_data_filename, poisoned_data_filename, bd_model_filename, prune_model_filename)
