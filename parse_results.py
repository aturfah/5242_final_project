from config import Config
from misc import load_model_history, clean_model_history, merge_results
import pandas as pd
from copy import deepcopy

def handle_fold_performance(fold_info):
    output = {}

    epochs = []
    for type_ in ["train", "valid", "test"]:
        output_key = "{}_{}_{}".format("{}", type_, "{}")
        temp_loss = []
        temp_acc = []
        for fold in fold_info:
            info = fold[type_]
            temp_loss.append(info[0])
            temp_acc.append(info[1])
            if len(epochs) < 100 / Config.validation_pct:
                epochs.append(fold["epochs"])

        output[output_key.format("{}", "loss")] = sum(temp_loss) / len(temp_loss)
        output[output_key.format("{}", "acc")] = sum(temp_acc) / len(temp_acc)

    output["avg_epochs"] = sum(epochs) / len(epochs)

    return output


def process_results_dictionary(res):
    output = {}
    res_keys = res.keys()

    for key_ in res_keys:
        if isinstance(res[key_], list):
            vals = handle_fold_performance(res[key_])
            for key2 in vals:
                output[key2.format(key_)] = vals[key2]
        else:
            if key_ == "architecture":
                output[key_] = Config.ARCHITECTURE_MAP.get(res[key_])
            elif key_ == "regularization":
                output[key_] = Config.REGULARIZATION_MAP.get(res[key_])
            elif key_ == "initializer":
                output[key_] = Config.INITIALIZATION_MAP.get(res[key_])
            else:
                output[key_] = res[key_]

    return output


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

    target_columns = list(performance_df.columns[0:4]) + ["avg_epochs"]
    target_columns = target_columns + ["^{}.*".format(dataset_name)]

    dataset_df = performance_df.filter(regex="|".join(target_columns))
    dataset_df["dataset"] = [dataset_label] * len(list_of_performance)

    return dataset_df


def prepare_fashn(list_of_performance):
    performance_df = pd.DataFrame(list_of_performance)
    performance_df.drop(columns=["model_name"], inplace=True)

    fashn_df = performance_df.filter(regex="|".join(list(performance_df.columns[0:4]) + ["avg_time"] + ["fashion.*"]))
    fashn_df = prepare_df(fashn_df, prefix="fashion_mnist_")
    fashn_df["dataset"] = ["Fashion MNIST"] * len(list_of_performance)

    return fashn_df


def prepare_mnist(list_of_performance):
    performance_df = pd.DataFrame(list_of_performance)
    performance_df.drop(columns=["model_name"], inplace=True)    

    mnist_df = performance_df.filter(regex="|".join(list(performance_df.columns[0:4]) + ["avg_time"] + ["^[^f][^a][^s][^h]"]))
    mnist_df = prepare_df(mnist_df, prefix="mnist_")
    mnist_df["dataset"] = ["MNIST"] * len(list_of_performance)

    return mnist_df


def prepare_cifar10(list_of_performance):
    performance_df = pd.DataFrame(list_of_performance)
    performance_df.drop(columns=["model_name"], inplace=True)    

    mnist_df = performance_df.filter(regex="|".join(list(performance_df.columns[0:4]) + ["avg_time"] + ["^[^f][^a][^s][^h]"]))
    mnist_df = prepare_df(mnist_df, prefix="cifar10_")
    mnist_df["dataset"] = ["CIFAR10"] * len(list_of_performance)

    return mnist_df


if __name__ == "__main__":
    base_model_results = None
    loop_cv_results = clean_model_history(load_model_history())

    for old_fname in Config.old_results_fnames:
        loop_cv_results = merge_results(loop_cv_results,
            clean_model_history(load_model_history(old_fname)))

    model_performance = []

    ## Get all model results in a Pandas DF
    for key in loop_cv_results:
        res = loop_cv_results[key]
        if res["model_name"] == "base_model":
            if base_model_results:
                raise RuntimeError("DUPLICATED BASE MODEL RESULTS")
            base_model_results = process_results_dictionary(res)
        else:
            model_performance.append(process_results_dictionary(res))

    for dataset_name in Config.DATASETS:
        print("\n\n{}".format(dataset_name))
        prepare_dataset(model_performance, dataset_name)
    
    raise RuntimeError("DOOT DOOT")

    performance_df = pd.DataFrame(model_performance)
    performance_df.drop(columns=["model_name"], inplace=True)

    ## Prepare Base Model results
    base_mnist = prepare_mnist([base_model_results])
    base_fashn = prepare_fashn([base_model_results])

    # Write for R analysis
    total_base = pd.concat([base_mnist, base_fashn])
    total_base.to_csv(Config.base_results_fname)




    ## Split across datasets & prepare them
    fashn_df = prepare_fashn(model_performance)
    mnist_df = prepare_mnist(model_performance)

    # Write this to CSV because I want to use R to do analysis
    total_df = pd.concat([mnist_df, fashn_df])
    total_df.to_csv(Config.proc_results_fname)