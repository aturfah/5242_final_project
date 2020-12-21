from config import Config
from config import MNIST, FASHION_MNIST, CIFAR10, KMNIST, K49
from misc import load_model_history, clean_model_history, merge_results, read_filenames_in_directory
import pandas as pd
from copy import deepcopy
from numpy import median

from pprint import pprint

def handle_fold_performance(fold_info):
    output = {}

    epochs = []
    for fold in fold_info:
        epochs.append(fold["epochs"])

    for type_ in ["train", "valid", "test"]:
        output_key = "{}_{}_{}".format("{}", type_, "{}")
        temp_acc = []
        for fold in fold_info:
            info = fold[type_]
            temp_acc.append(info[1])

        output[output_key.format("{}", "acc")] = sum(temp_acc) / len(temp_acc)

    output["{}_avg_epochs"] = sum(epochs) / len(epochs)
    output["{}_num_folds"] = len(fold_info)

    return output


def process_fold_performance(fold_info, fold_idx):
    output = {}
    fold = fold_info[fold_idx]
    for type_ in ["train", "valid", "test"]:
        output_key = "{}_{}_{}".format("{}", type_, "{}")
        output[output_key.format("{}", "acc")] =  fold[type_][1]

    output["{}_epochs"] = fold["epochs"]
    output["fold_idx"] = fold_idx

    return output


def process_results_dictionary(res):
    all_results = []
    res_keys = res.keys()

    for fold_idx in range(int(100 / Config.validation_pct)):
        output = {}
        for key_ in res_keys:
            if isinstance(res[key_], list):
                vals = process_fold_performance(res[key_], fold_idx)
                for key2 in vals:
                    output[key2.format(key_)] = vals[key2]
            else:
                if key_ == "architecture":
                    output[key_] = Config.ARCHITECTURE_MAP.get(res[key_])
                elif key_ == "regularization":
                    output[key_] = Config.REGULARIZATION_MAP.get(res[key_])
                elif key_ == "initializer":
                    output[key_] = Config.INITIALIZATION_MAP.get(res[key_])
                elif key_ == "optimizer":
                    output[key_] = Config.OPTIMIZER_MAP.get(res[key_])
                else:
                    output[key_] = res[key_]

        all_results.append(output)

    return all_results


def prepare_df(df_in, prefix=None):
    output = deepcopy(df_in)
    # Rename some columns
    if prefix is not None:
        new_columns = []
        for col in df_in.columns:
            if col.startswith(prefix):
                new_columns.append(col.replace(prefix, ""))
            else:
                new_columns.append(col)

        output.columns = new_columns

    return output


def prepare_dataset(list_of_performance, dataset_name):
    dataset_label = Config.DATASET_NAME_MAP.get(dataset_name, dataset_name)

    performance_df = pd.DataFrame(list_of_performance)
    performance_df.drop(columns=["model_name"], inplace=True)

    target_columns = list(performance_df.columns[0:4]) + ["fold_idx"]
    target_columns = target_columns + ["^{}.*".format(dataset_name)]

    dataset_df = performance_df.filter(regex="|".join(target_columns))
    dataset_df["dataset"] = [dataset_label] * len(list_of_performance)

    return dataset_df


def prepare_output_df(list_of_performance):
    list_of_df = []
    for dataset_name in Config.DATASETS:
        temp = prepare_dataset(list_of_performance, dataset_name)
        temp = prepare_df(temp, "{}_".format(dataset_name))
        list_of_df.append(temp)

    return pd.concat(list_of_df)


if __name__ == "__main__":
    base_model_results = None
    loop_cv_results = clean_model_history(load_model_history())

    results_pkls = read_filenames_in_directory(Config.old_results_dir)

    for old_fname in results_pkls:
        loop_cv_results = merge_results(loop_cv_results,
            clean_model_history(load_model_history(old_fname)))

    print(len(loop_cv_results.keys()))

    model_performance = []

    ## Get all model results in a Pandas DF
    for key in loop_cv_results:
        res = loop_cv_results[key]
        if res["model_name"] == "base_model":
            if base_model_results:
                raise RuntimeError("DUPLICATED BASE MODEL RESULTS")
            base_model_results = process_results_dictionary(res)
        else:
            model_performance.extend(process_results_dictionary(res))

    ## Prepare Base Model results & Write for R analysis
    prepare_output_df(base_model_results).to_csv(Config.base_results_fname)

    ## Write out model results for the rest of them
    prepare_output_df(model_performance).to_csv(Config.proc_results_fname)
