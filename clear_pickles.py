"""
If architectures change but you don't want to spend 30 minutes
refitting the base model
"""
import pickle

if __name__ == "__main__":
    pickle_files = ["results_cv.pkl", "results_cv2.pkl", "results_cv3.pkl", "results_cv4.pkl", "results_cv5.pkl"]

    models_to_keep = "base_model"

    for filename in pickle_files:
        data = None
        with open(filename, 'rb') as in_file:
            data = pickle.load(in_file)

        models_to_delete = []
        for key in data.keys():
            if key != models_to_keep:
                models_to_delete.append(key)

        for key in models_to_delete:
            del data[key]

        print(data.keys())
        pickle.dump(data, open(filename, 'wb'))