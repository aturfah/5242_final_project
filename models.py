import tensorflow as tf
import time

from config import Config
from misc import tb_callback_prepare, process_architecture, process_regularizer, pull_data, process_optimizer

from copy import deepcopy

def create_base_model(name=None, dataset=None):
    base_model = tf.keras.Sequential(name=name)
    base_model.add(tf.keras.layers.InputLayer(
        input_shape=Config().DATASET_IMAGE_SHAPE[dataset])
    )
    base_model.add(tf.keras.layers.Flatten())
    base_model.add(tf.keras.layers.Dense(16, activation='relu'))
    base_model.add(tf.keras.layers.Dense(16, activation='relu'))
    base_model.add(tf.keras.layers.Dense(Config().DATASET_PREDICTION_HEAD[dataset]))
    return base_model


def compile_base_model(model):
    model.compile("adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['sparse_categorical_accuracy'])


def create_custom_model(name, arch, reg, initializer, dataset):
    mod = tf.keras.Sequential(name=name)
    mod.add(tf.keras.layers.InputLayer(
        input_shape=Config().DATASET_IMAGE_SHAPE[dataset])
    )

    # Data Augmentation
    mod.add(tf.keras.layers.experimental.preprocessing.RandomRotation(factor=0.08))
    mod.add(tf.keras.layers.experimental.preprocessing.RandomTranslation((-0.1, 0.1), (-0.1, 0.1)))

    layers = process_architecture(arch, initializer)
    for layer_info in layers[:-1]:
        model_layer = layer_info["layer"]
        if model_layer is not None:
            mod.add(model_layer)

        if layer_info["regularize"]:
            regularizer_layers = process_regularizer(reg)
            [mod.add(l) for l in regularizer_layers]

    mod.add(layers[-1]["layer"])

    return mod


def generate_model_name(model_type):
    architecture, regularization, initializer, optimizer = model_type.split("!")

    architecture = architecture.replace("|", "-")
    architecture = architecture.replace(":", ".")

    return "_".join([architecture, regularization, initializer, optimizer])


def fit_and_evaluate_fold(model_type, dataset_name, fold_idx):
    """
    Estimate model performance with Cross-Validation

    Returns model performance on each fold
    """
    output = []

    if model_type == "base_model":
        create_model_function = create_base_model
        compile_model_function = compile_base_model
        early_stop = False
        reduce_lr = False
        model_name_base = model_type
        batch_size = 128
        num_epochs = 100
    else:
        batch_size = Config().DATASET_BATCH_SIZE[dataset_name]
        model_type = model_type.format(prediction_head=Config.DATASET_PREDICTION_HEAD[dataset_name])
        def create_model_function(name, dataset):
            architecture, regularization, initializer = model_type.split("!")[:3]
            return create_custom_model(name,
                architecture,
                regularization,
                initializer,
                dataset)

        early_stop = True
        reduce_lr = False
        num_epochs = Config().MAX_TRAIN_EPOCHS
        model_name_base = generate_model_name(model_type)
        compile_model_function = lambda x: x.compile(
            process_optimizer(model_type.split("!")[-1]),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['sparse_categorical_accuracy']
        )

    train_data, valid_data, test_data = pull_data(dataset_name)
    backup_models = []

    abs_start_time = time.time()

    #### FIT THE MODEL
    model_name = "{}_{}_{}".format(model_name_base, dataset_name, fold_idx)
    model_path = Config().logs_base_dir.joinpath(model_name).joinpath("model")
    print("#" * 30 + "\nFitting", model_name)

    model_fold = create_model_function(name=model_name, dataset=dataset_name)
    compile_model_function(model_fold)
    tensorboard_callbacks = tb_callback_prepare(model_name, early_stop, reduce_lr)

    model_fold.summary()

    train_fold = train_data[fold_idx]
    valid_fold = valid_data[fold_idx]

    start_time = time.time()

    try:
        model_fold = tf.keras.models.load_model(model_path)
        hist = []
        print("Successfully loaded model!")
    except Exception:
        hist = model_fold.fit(train_fold.shuffle(1024, reshuffle_each_iteration=True).batch(batch_size).prefetch(100),
                validation_data=valid_fold.shuffle(1024).batch(batch_size).prefetch(100),
                epochs=num_epochs,
                verbose=1,
                callbacks=[tensorboard_callbacks]).history["loss"]

    print("Took {}s ({}s total)\n".format(int(time.time() - start_time),
                                        int(time.time() - abs_start_time)))

    output.append({
        "epochs": len(hist),
        "train": model_fold.evaluate(train_fold.batch(batch_size)),
        "valid": model_fold.evaluate(valid_fold.batch(batch_size)),
        "test": model_fold.evaluate(test_data.batch(batch_size))
    })


    model_fold.save(model_path)

    return output


def fit_and_evaluate(model_type, dataset_name):
    output = []
    for fold_idx in range(int(100 / Config.validation_pct)):
        output.append(fit_and_evaluate_fold(model_type, dataset_name, fold_idx))

    return output
