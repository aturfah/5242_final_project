"""
Determine optimal learning rates for different models.
Adapted from the code linked in the paper
"""

import matplotlib
matplotlib.use('TkAgg')

from matplotlib import pyplot as plt
import math
from keras.callbacks import LambdaCallback
import keras.backend as K
import numpy as np

from config import Config
from config import MNIST, KMNIST, FASHION_MNIST, K49
from models import create_custom_model
from misc import pull_data
from models import process_optimizer
import tensorflow_datasets as tfds

import tensorflow as tf
# To prevent the Blas xGEMM error thing
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


class LRFinder:
    """
    Plots the change of the loss function of a Keras model when the learning rate is exponentially increasing.
    See for details:
    https://towardsdatascience.com/estimating-optimal-learning-rate-for-a-deep-neural-network-ce32f2556ce0
    """

    def __init__(self, model):
        self.model = model
        self.losses = []
        self.lrs = []
        self.best_loss = 1e9

    def on_batch_end(self, batch, logs):
        # Log the learning rate
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)

        # Log the loss
        loss = logs['loss']
        self.losses.append(loss)

        # Check whether the loss got too large or NaN
        if batch > 5 and (math.isnan(loss) or loss > self.best_loss * 4):
            self.model.stop_training = True
            return

        if loss < self.best_loss:
            self.best_loss = loss

        # Increase the learning rate for the next batch
        lr *= self.lr_mult
        K.set_value(self.model.optimizer.lr, lr)

    def find(self, x_train, y_train, start_lr, end_lr, batch_size=64, epochs=1, **kw_fit):
        # If x_train contains data for multiple inputs, use length of the first input.
        # Assumption: the first element in the list is single input; NOT a list of inputs.
        N = x_train[0].shape[0] if isinstance(x_train, list) else x_train.shape[0]

        # Compute number of batches and LR multiplier
        num_batches = epochs * N / batch_size
        self.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) / float(num_batches))
        # Save weights into a file
        initial_weights = self.model.get_weights()

        # Remember the original learning rate
        original_lr = K.get_value(self.model.optimizer.lr)

        # Set the initial learning rate
        K.set_value(self.model.optimizer.lr, start_lr)

        callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))

        self.model.fit(x_train, y_train,
                       batch_size=batch_size, epochs=epochs,
                       callbacks=[callback],
                       **kw_fit)

        # Restore the weights to the state before model fitting
        self.model.set_weights(initial_weights)

        # Restore the original learning rate
        K.set_value(self.model.optimizer.lr, original_lr)

    def find_generator(self, generator, start_lr, end_lr, epochs=1, steps_per_epoch=None, **kw_fit):
        if steps_per_epoch is None:
            try:
                steps_per_epoch = len(generator)
            except (ValueError, NotImplementedError) as e:
                raise e('`steps_per_epoch=None` is only valid for a'
                        ' generator based on the '
                        '`keras.utils.Sequence`'
                        ' class. Please specify `steps_per_epoch` '
                        'or use the `keras.utils.Sequence` class.')
        self.lr_mult = (float(end_lr) / float(start_lr)) ** (float(1) / float(epochs * steps_per_epoch))

        # Save weights into a file
        initial_weights = self.model.get_weights()

        # Remember the original learning rate
        original_lr = K.get_value(self.model.optimizer.lr)

        # Set the initial learning rate
        K.set_value(self.model.optimizer.lr, start_lr)

        callback = LambdaCallback(on_batch_end=lambda batch,
                                                      logs: self.on_batch_end(batch, logs))

        self.model.fit_generator(generator=generator,
                                 epochs=epochs,
                                 steps_per_epoch=steps_per_epoch,
                                 callbacks=[callback],
                                 **kw_fit)

        # Restore the weights to the state before model fitting
        self.model.set_weights(initial_weights)

        # Restore the original learning rate
        K.set_value(self.model.optimizer.lr, original_lr)

    def plot_loss(self, n_skip_beginning=10, n_skip_end=5, x_scale='log'):
        """
        Plots the loss.
        Parameters:
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
        """
        opt_title = {
            "adam": "Adam",
            "sgd": "SGD"
        }[optimizer]

        fmt_dataset = {
            MNIST: "MNIST",
            FASHION_MNIST: "Fashion-MNIST",
            KMNIST: "Kuzushiji-MNIST",
            K49: "Kuzushiji-49"
        }[dataset_name]

        plt.figure("{}_{}".format(optimizer, dataset_name))

        plt.title("LR Curve for {} on {}".format(opt_title, fmt_dataset))
        plt.ylabel("loss")
        plt.xlabel("learning rate (log scale)")
        plt.plot(self.lrs[n_skip_beginning:-n_skip_end], self.losses[n_skip_beginning:-n_skip_end])
        plt.xscale(x_scale)
        plt.show()

    def plot_loss_change(self, sma=1, n_skip_beginning=10, n_skip_end=5, y_lim=(-0.01, 0.01)):
        """
        Plots rate of change of the loss function.
        Parameters:
            sma - number of batches for simple moving average to smooth out the curve.
            n_skip_beginning - number of batches to skip on the left.
            n_skip_end - number of batches to skip on the right.
            y_lim - limits for the y axis.
        """
        derivatives = self.get_derivatives(sma)[n_skip_beginning:-n_skip_end]
        lrs = self.lrs[n_skip_beginning:-n_skip_end]
        plt.ylabel("rate of loss change")
        plt.xlabel("learning rate (log scale)")
        plt.plot(lrs, derivatives)
        plt.xscale('log')
        plt.ylim(y_lim)
        plt.show()

    def get_derivatives(self, sma):
        assert sma >= 1
        derivatives = [0] * sma
        for i in range(sma, len(self.lrs)):
            derivatives.append((self.losses[i] - self.losses[i - sma]) / sma)
        return derivatives

    def get_best_lr(self, sma, n_skip_beginning=10, n_skip_end=5):
        derivatives = self.get_derivatives(sma)
        best_der_idx = np.argmin(derivatives[n_skip_beginning:-n_skip_end])
        return self.lrs[n_skip_beginning:-n_skip_end][best_der_idx]

architecture = "c:32:3:1|c:32:3:1|p:3|r|c:64:3:1|c:64:3:1|p:3|r|c:64:3:1|c:64:1:1|c:{prediction_head}:1:1|g"
# architecture = "f|d:128|d:{prediction_head}"
regularization = "B"
initializer = "glorot_uniform"
optimizer = "adam"
# optimizer = "sgd"
model_type = "!".join([architecture, regularization, initializer, optimizer])
dataset_name = K49 # MNIST, KMNIST, FASHION_MNIST, K49

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
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['sparse_categorical_accuracy']
        )

train_data, valid_data, test_data = pull_data(dataset_name)
train_fold = train_data[0]
valid_fold = valid_data[0]

model = create_model_function(model_type, dataset_name)
compile_model_function(model)

model.summary()

labels = []
images = []
for item in tfds.as_numpy(train_fold):
    images.append(item[0])
    labels.append(item[1])

x_train = np.asarray(images)
y_train = np.asarray(labels)

# LR Finder Code
lr_finder = LRFinder(model)
lr_finder.find(x_train, y_train, start_lr=1e-6, end_lr=1, batch_size=128, epochs=1)
lr_finder.plot_loss(n_skip_beginning=20, n_skip_end=5)
