"""
If some parts of the config changes & you don't want to waste
time loading models just to have them be ignored later
"""
from misc import load_model_history, save_model_history, clean_model_history, merge_results
from misc import read_filenames_in_directory


if __name__ == "__main__":
    pickle_files = read_filenames_in_directory("test_finished_pickles")

    base_data = {}
    for filename in pickle_files:
        temp = clean_model_history(load_model_history(filename))
        base_data = merge_results(base_data, temp)

    for filename in pickle_files:
        save_model_history(base_data, filename)
