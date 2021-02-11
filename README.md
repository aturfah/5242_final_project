# STAT 5242 Final Project: Model Design for Deep Learning in Practice
Author: Ali Turfah
Link to Report [here](5242_final_report.pdf)

## Preliminaries
#### Package Installation
```
## Install requirements
pip install -r requirements.txt

## If the above does not work, install the packages manually
pip install tensorflow==2.4.0
pip install tensorflow-addons==0.11.2
pip install tensorflow-datasets==4.1.0
pip install matplotlib==3.3.3
```

#### Retrieving Kuzushiji-49 data
See the `kuzushiji_files/` directory for more information

## Quickstart
To train the models according to `config.py` the models and generate the output, just call `./run_pipeline.sh`

See the Analysis Scripts section for more information on generating the output results

## Specifying model configuration options
Model configuration options and the locations to save files are specified in `config.py`. The actual mapping to a `tf.keras` class happens in `misc.py` in the appropriate `process_<name>` function

## Training Scripts
- `fold_balance_script.py` Ensures that folds for dataset contain similar amounts of the classes
- `lr_finder_test.py` Generates the learning rate graph for a single model configuration
- `train_models.py` Trains one model over one fold specified by the config and saves to a `.pkl` file. Skips models already in the `.pkl` file. Should **not** be run directly except for debugging purposes
- `run_pipeline.sh` Runs the training script for a specified number of models (should be 6400 if training all models). 

## Analysis Scripts
- `parse_results.py` Reads in specified `.pkl` files and writes the model performance to two CSV files: one for the base model and another for the models specified by `config.py`
- `pull_base_model_csvs.py` This pulls down the training/validation loss/accuracy data from the tensorboard graphs. Make sure to have Tensorboard running before calling this script as it uses the Tensorboard API
- `generate_base_model_output.R` Generates the tables and plots for the base model from the CSV output by `parse_results.py` and `pull_base_model_csvs.py` for the graphs
- `generate_other_model_output.R` Generates the tables and plots for the models specified by the config from the CSV output by `parse_results.py`

## Misc Scripts
- `clear_pickles.py` Clears all results from the specified `.pkl` files except for the `base_model`. Helpful if training needs to be restarted
- `combine_pickles.py` Consolidates the results across all specified `.pkl` files and writes the combined results back to the read files. Helpful when running multiple versions of the `run_pipeline.sh` script in parallel

## Results Files Provided
- `results_cv.pkl` Contains model performance across all folds for all configurations specified in `config.py` as well as the base model )to be processed by `parse_results.py`)
- `finished_pickles/` Directory with all the files used to generate `results_cv.pkl`
- `base_csv/` Directory with the training/validation information for the  base model trained across the datasets over its folds (for use in `generate_base_model_output.R`)
- `images` and `lr_images` contain the image output from the R scripts and `lr_finder_test.py`
