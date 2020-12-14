"""
If some parts of the config changes & you don't want to waste
time loading models just to have them be ignored later
"""
from misc import load_model_history, save_model_history, clean_model_history, merge_results

if __name__ == "__main__":
    pickle_files = ["results_cv.pkl", "results_cv2.pkl", "results_cv3.pkl", "results_cv4.pkl", "results_cv5.pkl"]

    base_data = {}
    for filename in pickle_files:
        temp = clean_model_history(load_model_history(filename))
        base_data = merge_results(base_data, temp)

    for filename in pickle_files:
        save_model_history(base_data, filename)
