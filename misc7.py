import os
import shutil
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

import pickle as pkl

from config7 import Config
from misc import process_optimizer, tb_callback_prepare, process_architecture
from copy import deepcopy

def prepare_log_dir():
    logs_base_dir = Config().logs_base_dir
    shutil.rmtree(logs_base_dir, True)
    os.makedirs(logs_base_dir, exist_ok=False)


def fmt_img(image, label):
    """Scale image for 0 mean and 1 variance"""
    image = tf.cast(image, dtype=tf.float32)

    # 2D Images can be wonky
    if tf.keras.backend.ndim(image) == 2:
        image = tf.reshape(image, image.shape + (1,))

    return tf.image.per_image_standardization(image), label


def pull_data(dataset_name):
    if dataset_name != "k49":
        train_raw, valid_raw, test_raw = pull_data_tfds(dataset_name)
        fmt_lambda = lambda x: (fmt_img(x["image"], x["label"]))
    else:
        train_raw, valid_raw, test_raw = pull_k49_data()
        fmt_lambda = lambda x, y: (fmt_img(x, y))

    train = [x.map(fmt_lambda) for x in train_raw]
    valid = [x.map(fmt_lambda) for x in valid_raw]
    test = test_raw.map(fmt_lambda)

    return train, valid, test


def pull_k49_data():
    file_path = Config().KUZUSHIJI_FILE_PATH
    prefix = "k49"
    images_stub = str(file_path.joinpath("{}-{}-imgs.npz".format(prefix, "{}")))
    labels_stub = str(file_path.joinpath("{}-{}-labels.npz".format(prefix, "{}")))

    ## Load the data from file directory
    train_images = np.load(images_stub.format("train"))['arr_0']
    train_labels = np.load(labels_stub.format("train"))['arr_0']
    test_images = np.load(images_stub.format("test"))['arr_0']
    test_labels = np.load(labels_stub.format("test"))['arr_0']

    ## Prepare the 5 folds of cross-validation data
    train_data = []
    valid_data = []
    valid_pct = Config().validation_pct
    # Prepare the indices
    num_samples = train_images.shape[0]
    num_splits = int(100 / valid_pct)
    for idx in range(num_splits):
        lower_bound = int(num_samples * idx*valid_pct / 100)
        upper_bound = int(num_samples * (idx+1)*valid_pct / 100)

        # Prepare Validation set for this fold
        img_valid = train_images[lower_bound:upper_bound,:,:]
        label_valid = train_labels[lower_bound:upper_bound]

        # Prepare Training data for this fold
        img_train1 = train_images[:lower_bound,:,:]
        label_train1 = train_labels[:lower_bound]
        img_train2 = train_images[upper_bound:,:,:]
        label_train2 = train_labels[upper_bound:]
        img_train = np.concatenate((img_train1, img_train2))
        label_train = np.concatenate((label_train1, label_train2))

        train_data.append(tf.data.Dataset.from_tensor_slices((img_train, label_train)))
        valid_data.append(tf.data.Dataset.from_tensor_slices((img_valid, label_valid)))

    test_data = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

    return train_data, valid_data, test_data


def pull_data_tfds(dataset_name):
    valid_pct = Config().validation_pct
    test_raw = tfds.load(dataset_name, split='test')
    valid_raw = tfds.load(dataset_name, split=[
        f'train[{k}%:{k+valid_pct}%]' for k in range(0, 100, valid_pct)
    ])
    valid_raw = tfds.load(dataset_name, split=["train[:20%]"])
    train_raw = tfds.load(dataset_name, split=[
        f'train[:{k}%]+train[{k+valid_pct}%:]' for k in range(0, 100, valid_pct)
    ])
    train_raw = tfds.load(dataset_name, split=["train[20%:]"])

    return train_raw, valid_raw, test_raw


def process_architecture(arch_str, initializer):
    """
    Take in Architecture string and return list of tf.keras.layers
    with appropriate initializers and the right number of logits
    in the final layer.
    """
    arch_layers = arch_str.split("|")

    output = []

    for layer in arch_layers:
        temp = {}
        temp["regularize"] = False
        temp["layer"] = None

        layer_key = layer[0]
        if layer_key == 'f':
            temp["layer"] = tf.keras.layers.Flatten()
        elif layer_key == 'd':
            units = [int(x) for x in layer.split(":")[1:]][0]
            temp["layer"] = tf.keras.layers.Dense(units, activation='relu', kernel_initializer=initializer)
        elif layer_key == 'c':
            filters, k_size, stride = [int(x) for x in layer.split(":")[1:]]
            temp["layer"] = tf.keras.layers.Convolution2D(filters, k_size, stride,
                activation='relu', kernel_initializer=initializer, padding="same")
        elif layer_key == 'p':
            p_size = [int(x) for x in layer.split(":")[1:]][0]
            temp["layer"] = tf.keras.layers.MaxPool2D(pool_size=p_size,
                padding="same", strides=2)
        elif layer_key == 'g':
            temp["layer"] = tf.keras.layers.GlobalAveragePooling2D()
        elif layer_key == 'r':
            temp["regularize"] = True
        else:
            raise NotImplementedError("ONLY SUPPORTS FLATTEN, CONV, AND DENSE")

        output.append(temp)

    return output

def process_regularizer(reg_str):
    """Take in the regularizer string and
    return list of corresponding lambdas"""

    if not reg_str:
        return []

    output = []
    for char in reg_str:
        if char == 'D':
            output.append(tf.keras.layers.Dropout(Config.DROPOUT_PROB))
        elif char == 'B':
            output.append(tf.keras.layers.BatchNormalization())
        else:
            raise NotImplementedError("ONLY SUPPORTS DROPOUT & BATCHNORM")

    return output


def load_model_history(fname=Config.saved_results_fname):
    output = {}
    try:
        output = pkl.load(open(fname, 'rb'))
        print("#" * 30, "OPENED {}".format(fname))
    except Exception:
        pass

    return output


def save_model_history(model_hist):
    pkl.dump(model_hist, open(Config.saved_results_fname, 'wb'))


def clean_model_history(model_history):
    """
    Remove models/datasets with an epoch length of 0
    """
    datasets = Config.DATASETS
    invalid_model_datasets = []
    for model_type in model_history.keys():
        for dataset in datasets:
            dataset_results = model_history[model_type].get(dataset)

            if dataset_results is None:
                continue

            valid_results = True

            for dataset_fold_info in dataset_results:
                print(dataset_fold_info["epochs"])
                if dataset_fold_info["epochs"] == 0:
                    print("\tDOOT")
                    valid_results = False
                    break

            if not valid_results:
                print("INVALID", model_type, dataset)
                invalid_model_datasets.append(
                    (model_type, dataset)
                )

    for model, dataset in invalid_model_datasets:
        del model_history[model_type][dataset]

    return model_history
