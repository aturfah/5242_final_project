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

    # DATASETS = (K49, )
    DATASETS = (MNIST, FASHION_MNIST, KMNIST, K49)

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
        "f|d:16|r|d:16|d:{prediction_head}", # Dumb linear models
        "c:32:3:1|p:3|r|c:64:3:1|p:3|r|c:64:3:1|c:64:1:1|c:{prediction_head}:1:1|g", # Model C
        "c:32:3:2|r|c:64:3:2|r|c:64:3:1|c:64:1:1|c:{prediction_head}:1:1|g", # Strided CNN-C
        "c:32:3:1|c:32:3:1|p:3|r|c:64:3:1|c:64:3:1|p:3|r|c:64:3:1|c:64:1:1|c:{prediction_head}:1:1|g", # ConvPool-CNN-C
        "c:32:3:1|c:32:3:2|r|c:64:3:1|c:64:3:2|r|c:64:3:1|c:64:1:1|c:{prediction_head}:1:1|g", # All-CNN-C
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
        "glorot_uniform", # Default
        "glorot_normal",
        'random_normal',
        'random_uniform'
    ]

    # Optimizers for model fitting
    model_opt = [
        "adam", 
        "swa",
        "rectified_adam",
        "sgd"
    ]

    ### Stuff for writing models to file
    saved_results_fname = "results_cv.pkl"
    saved_results_buffer = 1

    # old_results_dir = "finished_pickles"
    old_results_dir = "test_finished_pickles"

    ### For generate_results.py
    proc_results_fname = "proc_results.csv"
    base_results_fname = "base_results.csv"

    ARCHITECTURE_MAP = {
        model_arch[0]: "2FC",
        model_arch[1]: "Model C",
        model_arch[2]: "Strided CNN",
        model_arch[3]: "ConvPool CNN",
        model_arch[4]: "All CNN"
    }
    REGULARIZATION_MAP = {
        model_regularization_layer[0]: "Dropout",
        model_regularization_layer[1]: "BatchNorm",
        model_regularization_layer[2]: "Dropout / BatchNorm",
        model_regularization_layer[3]: "BatchNorm / Dropout"
    }
    INITIALIZATION_MAP = {
        model_init[0]: "Glorot Uniform",
        model_init[1]: "Glorot Normal",
        model_init[2]: "Random Normal",
        model_init[3]: "Random Uniform"
    }
    OPTIMIZER_MAP = {
        model_opt[0]: "Adam",
        model_opt[1]: "SGD-SWA",
        model_opt[2]: "RAdam",
        model_opt[3]: "SGD"
    }
    DATASET_NAME_MAP = {
        MNIST: "MNIST",
        FASHION_MNIST: "Fashion-MNIST",
        CIFAR10: "CIFAR-10",
        KMNIST: "Kuzushiji-MNIST",
        K49: "Kuzushiji-49"
    }
