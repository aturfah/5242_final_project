from pathlib import Path

MNIST = "mnist"
FASHION_MNIST = "fashion_mnist"
CIFAR10 = "cifar10"
KMNIST = "kmnist"
K49 = "k49"

class Config():
    validation_pct = 20
    logs_base_dir = Path("logs/")
    if not logs_base_dir.exists():
        logs_base_dir.mkdir()

    DATASETS = (K49, )
    # DATASETS = (MNIST, FASHION_MNIST, CIFAR10, KMNIST, K49)

    DATASET_IMAGE_SHAPE = {
        MNIST: (28, 28, 1),
        FASHION_MNIST: (28, 28, 1),
        CIFAR10: (32, 32, 3),
        KMNIST: (28, 28, 1),
        K49: (28, 28, 1)
    }

    DATASET_PREDICTION_HEAD = {
        MNIST: 10,
        FASHION_MNIST: 10,
        CIFAR10: 10,
        KMNIST: 10,
        K49: 49
    }

    KUZUSHIJI_FILE_PATH = Path("kuzushiji_files/")

    SEED = 2509
    DROPOUT_PROB = 0.5
    DATASET_BATCH_SIZE = {
        MNIST: 384,
        FASHION_MNIST: 384,
        CIFAR10: 384,
        KMNIST: 384,
        K49: 384
    }
    MAX_TRAIN_EPOCHS = 200

    """
    Model layers and parameters
    - | divides layers
    - f is Flatten Layer
    - d:# is Dense layer with # Units
    - c:#:@:$ is convolution with
        - # filters
        - @ kernel size
        - $ stride
    - p:# is a MaxPool with # Window and Stride 2
    - g is a global averaging layer
    """
    model_arch = [
        "c:32:3:1|c:32:3:2|r|c:64:3:1|c:64:3:2|r|c:64:3:1|c:64:1:1|c:{prediction_head}:1:1|g", # All-CNN-C
        "c:32:3:1|c:32:3:1|p:3|r|c:64:3:1|c:64:3:1|p:3|r|c:64:3:1|c:64:1:1|c:{prediction_head}:1:1|g", # ConvPool-CNN-C
        "c:32:3:2|r|c:64:3:2|r|c:64:3:1|c:64:1:1|c:{prediction_head}:1:1|g", # Strided CNN-C
        "c:32:3:1|p:3|r|c:64:3:1|p:3|r|c:64:3:1|c:64:1:1|c:{prediction_head}:1:1|g", # Model C
        "f|d:16|r|d:16|d:{prediction_head}", # Dumb linear models
    ]

    # D is Dropout, B is BatchNorm
    model_regularization_layer = [
        "D",
        "B",
        "DB",
        "BD"
    ]

    # Initializers for layer params
    model_init = [
        # "glorot_uniform", # Default
        # "glorot_normal",
        'random_normal',
        'random_uniform'
    ]

    # Optimizers for model fitting
    model_opt = [
        # "adam", 
        # "swa",
        # "rectified_adam",
        "sgd"
    ]

    ### Stuff for writing models to file
    saved_results_fname = "results_sgd_k49_random.pkl"
    saved_results_buffer = 1

    old_results_fnames = ["results_cv{}.pkl".format(idx) for idx in range(1, 11)] +\
         ["finished_pickles/results_cv.pkl"] + ["finished_pickles/results_cv{}.pkl".format(idx) for idx in range(1, 15)]
