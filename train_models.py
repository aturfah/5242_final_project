import tensorflow as tf

tf.get_logger().setLevel("ERROR")

# To prevent the Blas xGEMM error thing
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Misc things
import numpy as np
import gc

# My stuff
from models import fit_and_evaluate
from misc import load_model_history, save_model_history
from config import Config
from config import MNIST, FASHION_MNIST, CIFAR10, KMNIST, K49

print("TF version: ",tf.__version__)
print("Keras version:",tf.keras.__version__)

tf.random.set_seed(Config.SEED)

# prepare_log_dir()

### Training Loop for Base Model
loop_cv_results = load_model_history()

if "base_model" not in loop_cv_results:
    loop_cv_results["base_model"] = {
        "model_name": "base_model",
        "architecture": None,
        "regularization": None,
        "initializer": None,
        "optimizer": None,
        MNIST: fit_and_evaluate("base_model", MNIST),
        FASHION_MNIST: fit_and_evaluate("base_model", FASHION_MNIST),
        CIFAR10: fit_and_evaluate("base_model", CIFAR10),
        KMNIST: fit_and_evaluate("base_model", KMNIST),
        K49: fit_and_evaluate("base_model", K49)
    }
    save_model_history(loop_cv_results)

### Training Loop for Other Models
total_models = 1 + len(Config.model_arch) * \
    len(Config.model_regularization_layer) * \
        len(Config.model_init) * \
            len(Config.model_opt) * \
                len(Config.DATASETS)
counter = 1
for initializer in Config.model_init:
    for regularization in Config.model_regularization_layer:
        for optimizer in Config.model_opt:
            for architecture in Config.model_arch:
                for dataset in Config.DATASETS:
                    # Check if results have already been processesd
                    counter += 1
                    model_name = "!".join([architecture, regularization, initializer, optimizer])
                    print("\n\nModel #{} / {}\n".format(counter, total_models))

                    if model_name in loop_cv_results:
                        if dataset in loop_cv_results[model_name]:
                            print("Already have results for this model on {}!".format(dataset))
                            continue
                    else:
                        loop_cv_results[model_name] = {
                            "model_name": model_name,
                            "architecture": architecture,
                            "regularization": regularization,
                            "initializer": initializer,
                            "optimizer": optimizer
                        }

                    print(dataset, loop_cv_results[model_name].keys())
                    loop_cv_results[model_name][dataset] = fit_and_evaluate(model_name, dataset)

                    # Write results to file
                    if counter % Config.saved_results_buffer == 0:
                        print("\n\nWRITING RESULTS TO PKL\n\n")
                        save_model_history(loop_cv_results)

                        # Out of Memory Issues when fitting a lot of these bad boys
                        tf.keras.backend.clear_session()
                        tf.compat.v1.reset_default_graph()
                        gc.collect()

                    # Abort and restart each step; preserve sanity with RAM usage
                    raise RuntimeError("MAXED OUT, TRY AGAIN LATER")


save_model_history(loop_cv_results)
print(len(loop_cv_results))