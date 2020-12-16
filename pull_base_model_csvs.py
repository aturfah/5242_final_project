"""
Pull data for base model from Tensorboard.


Much faster than manually downloading the CSVs.
"""

import requests
from config import Config 

from os import mkdir
from os.path import exists

if __name__ == "__main__":
    LOGS_DIR = "base_csv"
    if not exists(LOGS_DIR):
        mkdir(LOGS_DIR)

    TENSORBOARD_URL = "http://localhost:6006"
    BASE_URL = "{}/{}".format(TENSORBOARD_URL, "data/plugin/scalars/scalars?tag={TAG}&run={RUN_ID}/{MODE}&format=csv")
    TAGS = ["epoch_loss", "epoch_sparse_categorical_accuracy"]
    RUN_ID = "base_model_{dataset}_{fold}"
    MODES = ["train", "validation"]
    DATASETS = Config.DATASETS
    FOLDS = [x for x in range(int(100/Config.validation_pct))]

    OUT_FILENAME = "{}/{}".format(LOGS_DIR, "base_model_{mode}_{dataset}_{fold}_{tag}.csv")

    for dataset in DATASETS:
        for fold in FOLDS:
            run_id = RUN_ID.format(dataset=dataset, fold=fold)
            for mode in MODES:
                for tag in TAGS:
                    url = BASE_URL.format(TAG=tag, RUN_ID=run_id, MODE=mode)
                    r = requests.get(url)
                    fname = OUT_FILENAME.format(mode=mode, dataset=dataset, fold=fold, tag=tag)
                    with open(fname, 'wb') as out_file:
                        out_file.write(r.content)
