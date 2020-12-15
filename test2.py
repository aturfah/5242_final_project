import tensorflow as tf
import numpy as np
from config import Config
from models import create_custom_model
from misc import pull_data, tb_callback_prepare
from models import process_optimizer
import tensorflow_datasets as tfds

tf.random.set_seed(10)

# To prevent the Blas xGEMM error thing
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


architecture = "c:32:3:1|c:32:3:2|r|c:64:3:1|c:64:3:2|r|c:64:3:1|c:64:1:1|c:{prediction_head}:1:1|g"
regularization = "D"
initializer = "glorot_uniform"
optimizer = "adam"
model_type = "!".join([architecture, regularization, initializer, optimizer])
dataset_name = "cifar10"


model_type = model_type.format(prediction_head=Config.DATASET_PREDICTION_HEAD[dataset_name])


def create_model_function(name, dataset):
    architecture, regularization, initializer = model_type.split("!")[:3]
    return create_custom_model("doot",
        architecture,
        regularization,
        initializer,
        dataset)

compile_model_function = lambda x: x.compile(
            process_optimizer(model_type.split("!")[-1]),
            loss='sparse_categorical_crossentropy',
            metrics=['sparse_categorical_accuracy']
        )

train_data, valid_data, test_data = pull_data(dataset_name)

preprocess = tf.keras.Sequential([
   tf.keras.layers.experimental.preprocessing.RandomFlip(mode="horizontal"),
   tf.keras.layers.experimental.preprocessing.RandomRotation(factor=(-0.09, 0.09))
])

tensorboard_callbacks = tb_callback_prepare(model_type, True)

output = []
for idx in range(len(train_data)):
    train_fold = train_data[idx]
    valid_fold = valid_data[idx]

    aug_ds = train_fold.batch(256).map(lambda x, y: (preprocess(x, training=True), y))
    aug_ds = train_fold.batch(256)

    model = create_model_function(model_type, dataset_name)
    compile_model_function(model)

    history = model.fit(aug_ds,
        validation_data=valid_fold.batch(256),
        epochs=100, callbacks=[tensorboard_callbacks])

    output.append({
            "train": model.evaluate(aug_ds),
            "valid": model.evaluate(valid_fold.batch(256)),
            "test": model.evaluate(test_data.batch(256))
        })

for something in output:
    print("\n\n")
    print(something)