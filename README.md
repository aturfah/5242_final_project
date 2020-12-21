# STAT 5242 Final Project: Model Design for Deep Learning in Practice
Ali Turfah

## Installation
```
## Install requirements
pip install -r requirements.txt

## If the above does not work, install the packages manually
pip install tensorflow==2.4.0
pip install tensorflow-addons==0.11.2
pip install tensorflow-datasets==4.1.0
pip install matplotlib==3.3.3
```

## Quickstart
To use the results of already trained models, run the Analysis scripts based on `results_cv.pkl` (which is the consolidated version of the the files stored in `finished_pickles/`)

To train the models, just call `./run_pipeline.sh` which will begin training models according to `config.py`

## Specifying model configuration options
Model configuration options and the locations to save files are specified in `config.py`. The actual mapping to a `tf.keras` class happens in `misc.py` in the appropriate `process_<name>` function.

## Retrieving Kuzushiji-MNIST and Kuzushiji-49 data
See the `kuzushiji_files/` directory for more information.

## Training Scripts
- `fold_balance_script.py` Ensures that folds for dataset contain similar amounts of the classes
- `lr_finder_test.py` Generates the learning rate graph for a single model configuration
- `train_models.py` Trains one model over one fold specified by the config and saves to a `.pkl` file. Should **not** be run directly except for debugging purposes
- `run_pipeline.sh` Runs the training script for a specified number of models (should be 6400 if training all models). 

## Analysis Scripts
- `parse_results.py` Reads in specified `.pkl` files and writes the model performance to two CSV files: one for the base model and another for the models specified by `config.py`
- `generate_base_model_output.R` Generates the tables and plots for the base model from the CSV output by `parse_results.py`
- `generate_other_model_output.R` Generates the tables and plots for the models specified by the config from the CSV output by `parse_results.py`

## Misc Scripts
- `clear_pickles.py` Clears all results from the specified `.pkl` files except for the `base_model`. Helpful if training needs to be restarted
- `combine_pickles.py` Consolidates the results across all specified `.pkl` files and writes the combined results back to the read files. Helpful when running multiple versions of the `run_pipeline.sh` script in parallel.