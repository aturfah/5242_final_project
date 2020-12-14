import os
import shutil
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
import numpy as np

import pickle as pkl

from config import Config
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
    train_raw = tfds.load(dataset_name, split=[
        f'train[:{k}%]+train[{k+valid_pct}%:]' for k in range(0, 100, valid_pct)
    ])

    return train_raw, valid_raw, test_raw


def tb_callback_prepare(model_name, early_stop=False, reduce_lr=False):
    logs_base_dir = Config().logs_base_dir

    new_dir = logs_base_dir.joinpath(model_name)
    if not new_dir.exists():
        # new_dir.rmdir()
        new_dir.mkdir()

    ## The Vanilla Tensorboard Callback
    tbcb = tf.keras.callbacks.TensorBoard(
        log_dir=new_dir, histogram_freq=1)

    output = [tbcb]

    ## Early Stopping
    if early_stop:
        escb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        output.append(escb)

    return output


def parse_layer(layer, initializer, final_layer=False):
    """
    Map the layer key to the appropriate layer definition
    """
    temp = {}
    temp["regularize"] = False
    temp["layer"] = None

    if not final_layer:
        activation = "relu"
    else:
        activation = None

    layer_key = layer[0]
    if layer_key == 'f':
        temp["layer"] = tf.keras.layers.Flatten()
    elif layer_key == 'd':
        units = [int(x) for x in layer.split(":")[1:]][0]
        temp["layer"] = tf.keras.layers.Dense(units, activation=activation, kernel_initializer=initializer)
    elif layer_key == 'c':
        filters, k_size, stride = [int(x) for x in layer.split(":")[1:]]
        temp["layer"] = tf.keras.layers.Convolution2D(filters, k_size, stride,
            activation=activation, kernel_initializer=initializer, padding="same")
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

    return temp


def process_architecture(arch_str, initializer):
    """
    Take in Architecture string and return list of tf.keras.layers
    with appropriate initializers and the right number of logits
    in the final layer.
    """
    arch_layers = arch_str.split("|")

    output = []

    for layer in arch_layers[:-1]:
        temp = parse_layer(layer, initializer)
        output.append(temp)

    # Do the final layer separately
    output.append(parse_layer(arch_layers[-1], initializer, final_layer=True))

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


def process_optimizer(opt_str):
    optimizer_dict = {
        "adam": tf.keras.optimizers.Adam(),
        "adam2": tf.keras.optimizers.Adam(learning_rate=0.01),
        "adam4": tf.keras.optimizers.Adam(learning_rate=1e-4),
        "adam5": tf.keras.optimizers.Adam(learning_rate=1e-5),
        "adagrad": tf.keras.optimizers.Adagrad(),
        "rmsprop": tf.keras.optimizers.RMSprop(),
        "sgd": tf.keras.optimizers.SGD(momentum=0.9),
        "swa": tfa.optimizers.SWA(tf.keras.optimizers.SGD(momentum=0.9)),
        "rectified_adam": tfa.optimizers.RectifiedAdam(min_lr=1e-10)
    }

    if opt_str not in optimizer_dict:
        raise RuntimeError("Invalid optimizer: {}".format(opt_str))

    return optimizer_dict[opt_str]


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
        if model_type == "base_model":
            continue

        for dataset in datasets:
            dataset_results = model_history[model_type].get(dataset)

            if dataset_results is None:
                continue

            valid_results = True

            for dataset_fold_info in dataset_results:
                if dataset_fold_info["epochs"] == 0:
                    valid_results = False
                    break

            if not valid_results:
                print("INVALID", model_type, dataset)
                invalid_model_datasets.append(
                    (model_type, dataset)
                )

    for model, dataset in invalid_model_datasets:
        del model_history[model][dataset]

    return model_history


def merge_results(base_dict, target_dict):
    output = deepcopy(base_dict)

    base_keys = set(base_dict.keys())
    tgt_keys = set(target_dict.keys())

    shared_keys = base_keys.intersection(tgt_keys)

    for key in tgt_keys:
        if key in shared_keys:
            print("Ignoring: {}".format(key))
        output[key] = target_dict[key]

    print(len(base_keys))
    print(len(tgt_keys))
    print(len(output.keys()))

    return output