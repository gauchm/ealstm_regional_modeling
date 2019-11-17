# The Proper Care and Feeding of CAMELS: How Limited Training Data Affects Streamflow Prediction

This repository accompanies the paper *The Proper Care and Feeding of CAMELS: How Limited Training Data Affects Streamflow Prediction* (Gauch et al., 2019).

It is built upon the code from https://github.com/kratzert/ealstm_regional_modeling.

## Content of the repository

- `main_gridEvaluation.py` Main python file used to generate bash scripts (more precisely, SLURM submission scripts) that train and evaluate XGBoost and EA-LSTM models on different amounts of training data.
- `main.py` Main python file used for training and evaluating EA-LSTM models
- `main_xgboost.py` Main python file used for training and evaluating XGBoost models (including random parameter search)
- `data/` contains the list of basins (USGS gauge ids) considered in our study, and shapefiles of the continental US ([source](https://github.com/matplotlib/basemap/tree/master/examples)), used for plotting.
- `papercode/` contains the entire code (besides the `main*.py` files in the root directory)
- `notebooks/` contains the notebook that guides through the results of our study (as well as the notebooks from the original Kratzert et al. paper).
    - `notebooks/performance_gridEvaluation.ipynb`: This notebook evaluates and compares the results of XGBoost and EA-LSTMs trained on different amounts of training data.
    - `notebooks/performance.ipynb`: This notebooks evaluates and compares the LSTM architechtures from Kratzert et al.
    - `notebooks/ranking.ipynb`: This notebooks evaluates feature rankings and model robustness of the LSTM architechtures from Kratzert et al.
    - `notebooks/embedding.ipynb`: This notebooks analyzes the LSTM catchment embeddings from Kratzert et al.

## Setup to run the code locally

Download this repository either as zip-file or clone it to your local file system by running

```
git clone git@github.com:gauchm/ealstm_regional_modeling.git
```

### Setting up the Python environment
Within this repository, we provide two environment files (`environment_cpu.yml` and `environment_gpu.yml`) that can be used with Anaconda or Miniconda to create an environment with all needed packages.

Simply run

```
conda env create -f environment_cpu.yml
```
for the cpu-only version. Or run

```
conda env create -f environment_gpu.yml
```
if you have a CUDA-capable NVIDIA GPU. This is recommended if you want to train/evaluate the LSTM on your machine, but not strictly necessary. 

In addition, you will have to install XGBoost from source (the current version on conda, 0.90, has a bug that affects training with a custom objective):
```
conda activate ealstm
git clone https://github.com/dmlc/xgboost --recursive
git checkout 96cd7ec2bbdec1addf81b1ca2adb13c9155e32f3  # this is the version we used in our study
cd xgboost; mkdir build; cd build;
cmake ..
make -j4
cd ../python-package
python setup.py install
```

## Data

### Required Downloads

First, you need the CAMELS data set to run any of your code. This data set can be downloaded for free here:

- [CAMELS: Catchment Attributes and Meteorology for Large-sample Studies - Dataset Downloads](https://ral.ucar.edu/solutions/products/camels) Make sure to download the `CAMELS time series meteorology, observed flow, meta data (.zip)` file, as well as the `CAMELS Attributes (.zip)`. Extract the data set on your file system and make sure to put the attribute folder (`camels_attributes_v2.0`) inside the CAMELS main directory.

However, we trained our models with an updated version of the Maurer forcing data that is not yet officially published (CAMELS data set will be updated soon). The updated Maurer forcings contain daily minimum and maximum temperature, while the original Maurer data included in the CAMELS data set only include daily mean temperature. You can find the updated forcings temporarily here:

- [Updated Maurer forcing with daily minimum and maximum temperature](https://www.hydroshare.org/resource/17c896843cf940339c3c3496d0c1c077/)

Download and extract the updated forcing into the `basin_mean_forcing` folder of the CAMELS data set and do not rename it (name should be `maurer_extended`).

### Optional Downloads

The pre-trained XGBoost and EA-LSTM models, the predictions, and the SLURM submission scripts from our study are available for download here:

- [Pre-trained models, predictions, scripts](https://doi.org/10.5281/zenodo.3543549)

To download pre-trained models and simulations, and the physically-based benchmark models from the Kratzet et al. paper, use the following links:

- [Original pre-trained models](http://www.hydroshare.org/resource/83ea5312635e44dc824eeb99eda12f06)
- [Physically-based benchmark models](http://www.hydroshare.org/resource/474ecc37e7db45baa425cdb4fc1b61e1)


## Running locally

For training or evaluating any of the LSTM models a CUDA-capable NVIDIA GPU is recommended but not strictly necessary. Since we only train/use LSTM-based models, a strong multi-core CPU will work as well.

Before starting, make sure you have activated the conda environment.

```
conda activate ealstm
```

### Training models

#### LSTMs
To train an LSTM model, run the following line of code from the terminal

```
python main.py train --camels_root /path/to/CAMELS
```
This would train a single EA-LSTM model with a randomly generated seed using the basin average NSE as loss function and store the results under `runs/`. Additionally the following options can be passed:

- `--seed NUMBER` Train a model using a fixed random seed
- `--cache_data True` Load the entire training data into memory. This will speed up training but requires approximately 50GB of RAM.
- `--num_workers NUMBER` Defines the number of parallel threads that will load and preprocess inputs.
- `--train_start` This date (formatted as `ddmmyyyy`will be used as the training start date.
- `--train_start` This date (formatted as `ddmmyyyy`will be used as the training enddate.
- `--no_static True` If passed, will train a standard LSTM without static features. If this is not desired, don't pass `False` but instead remove the argument entirely.
- `--concat_static True` If passed, will train a standard LSTM where the catchment attributes as concatenated at each time step to the meteorological inputs. If this is not desired, don't pass `False` but instead remove the argument entirely.
- `--use_mse True` If passed, will train the model using the mean squared error as loss function. If this is not desired, don't pass `False` but instead remove the argument entirely.
- `--run_dir_base` If passed, will store training data and results in a subfolder of this folder. Default is `runs/`
- `--run_name` If passed, will store training data and results in `{run_dir_base}/{run_name}`. By default, a name is generated based on the current date and time.
- `--basins` If passed, will only use these basins during training (evaluation automatically only uses the basins that were used in training). Pass multiple basins separated by spaces. Default is all 531 basins.

#### XGBoost
To train an XGBoost model, run the following line of code from the terminal

```
python main_xgboost.py train --camels_root /path/to/CAMELS
```
This would train a single XGBoost model with a randomly generated seed using MSE as objective and store the results under `runs/`. Additionally the following options can be passed:

- `--seed NUMBER` Train a model using a fixed random seed
- `--num_workers NUMBER` Defines the number of parallel threads that will load and preprocess inputs.
- `--train_start` This date (formatted as `ddmmyyyy`will be used as the training start date.
- `--train_start` This date (formatted as `ddmmyyyy`will be used as the training enddate.
- `--no_static True` If passed, will train a model without static features. If this is not desired, don't pass `False` but instead remove the argument entirely.
- `--use_mse` If passed, will train the model using MSE as objective. If this is not desired, remove the argument entirely.
- `--model_dir` If passed, will train an XGBoost model using the model parameters from the run in this folder (pass the directory that contains the `model.pkl` file). If not passed, training will include a random search for suitable parameters.
- `--run_dir_base` If passed, will store training data and results in a subfolder of this folder. Default is `runs/`
- `--run_name` If passed, will store training data and results in `{run_dir_base}/{run_name}`. By default, a name is generated based on the current date and time.
- `--basins` If passed, will only use these basins during training (evaluation automatically only uses the basins that were used in training). Pass multiple basins separated by spaces. Default is all 531 basins.

### Running inference and evaluating trained models

Once training is finished, you can use the models to run inference and generate predictions for the test period.
This will calculate the discharge simulation for the validation period and store the results alongside the observed discharge for all basins that were used during training in a pickle file. The pickle file is stored in the main directory of the model run.

After inference, you can run the notebook in `notebooks/performance_gridEvaluation.ipynb` to evaluate the predictions' accuracy.

#### LSTMs
To generate predictions with an LSTM model, run the following line of code from the terminal.

```
python main.py evaluate --camels_root /path/to/CAMELS --run_dir path/to/model_run
```

#### XGBoost
To generate predictions with an XGBoost model, run the following line of code from the terminal.

```
python main_xgboost.py evaluate --camels_root /path/to/CAMELS --run_dir path/to/model_run
```


### Creating scripts to train and evaluate on varying amounts of data
To create SLURM submission scripts that automatically train and evaluate XGBoost and EA-LSTM models on varying amounts of training data, run the following line of code from the terminal.

```
python main_gridEvaluation.py --camels_root /path/to/CAMELS
```
Additionally, the following options can be passed:

- `--num_workers_ealstm` Use this option to determine the number of workers used for EA-LSTM training. Default is 12.
- `--num_workers_xgb` Use this option to determine the number of workers used for EA-LSTM training. Default is 20.
- `--use_mse` Provide this option if you want to use NSE as objective and loss function in XGBoost and EA-LSTM training.
- `--user` Use this option to set the email address that SLURM job failure notifications will be sent to.
- `--use_params` Use this option to reuse XGBoost parameters from the model in the specified directory, rather than performing a parameter search.

The script will generate SLURM submission scripts in a folder `run_grid_ddmm_hhmm/`, which you can either execute as normal bash scripts (note that you will need to make them executable through `chmod +x path/to/script.sbatch`) or submit to a SLURM scheduler via `sbatch path/to/script.sbatch` (note that you will need to adapt the `account` submission parameter).
The following scripts will be generated:

- There will be one EA-LSTM training script (running `main.py`) for each combination of basins, training years, and seed (`run_ealstm_{train_start}_{train_end}_basinsample{number_of_basins}_{id_of_basinsample}_seed{seed}.sbatch`).
- For XGBoost, there will be two types of scripts (both running `main_xgboost.py`):
  - One script for to find suitable parameters in a random search (`run_xgb_param_search_{train_start}_{train_end}_basinsample{number_of_basins}_{id_of_basinsample}_seed111.sbatch`). 
  - Additionally, there will be one XGBoost training script for each combination of basins, training years, and seed that use the parameters from the above parameter search to train models (`run_xgb_train_{train_start}_{train_end}_basinsample{number_of_basins}_{id_of_basinsample}_seed{seed}.sbatch`). Because these scripts need the parameters from the `xgb_param_search` runs, you can only execute them after completion of the parameter search.

### Evaluating robustness (LSTMs only)

To evaluate the LSTM model robustness against noise of the static input features run the following line of code from the terminal.

```
python main.py eval_robustness --camels_root /path/to/CAMELS --run_dir path/to/model_run
```

This will run 265,500 model evaluations (10 levels of added random noise and 50 repetitions per noise level for 531 basins). This evaluations is only implemented for our EA-LSTM. Therefore, make sure that the `model_run` folder contains the results of training an EA-LSTM.

### Running notebooks

In your terminal, go to the project folder and start a jupyter notebook server by running

```
jupyter notebook
```


## Citation

If you use any of this code in your experiments, please make sure to cite our paper as well as the following original publication by Kratzert et al.:

```
Gauch, M., Mai, J., Lin, J., "The Proper Care and Feeding of CAMELS: How Limited Training Data Affects Streamflow Prediction Models" (2019)

Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G., "Benchmarking
a Catchment-Aware Long Short-Term Memory Network (LSTM) for Large-Scale Hydrological Modeling".
submitted to Hydrol. Earth Syst. Sci. Discussions (2019)
```

## License
[Apache License 2.0](https://github.com/gauchm/ealstm_regional_modeling/blob/master/LICENSE)

## License of the updated Maurer forcings and our pre-trained models
The CAMELS data set only allows non-commercial use. Thus, our pre-trained models and the updated Maurer forcings underlie the same [TERMS OF USE](https://www2.ucar.edu/terms-of-use) as the CAMELS data set. 

