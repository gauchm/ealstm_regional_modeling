"""
Code to train and evaluate Random Forest models.
Adapted from https://github.com/kratzert/ealstm_regional_modeling/blob/master/main.py
"""

import argparse
import json
import pickle
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path, PosixPath
from typing import Dict, List, Tuple

import numpy as np
import scipy as sp
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm

from papercode.datasets import CamelsH5, CamelsTXT
from papercode.datautils import (add_camels_attributes, load_attributes,
                                 rescale_features)
from papercode.metrics import calc_nse
from papercode.nseloss import NSELoss, NSEObjective
from papercode.utils import create_h5_files, get_basin_list

###########
# Globals #
###########

# fixed settings for all experiments
GLOBAL_SETTINGS = {
    # parameters for RandomSearchCVs:
    'param_dist': {
        'max_depth': sp.stats.randint(2, 20),
        'min_samples_split': sp.stats.randint(2, 20), 
        'min_samples_leaf': sp.stats.randint(1, 20), 
    },
    'n_iter': 500,
    'n_estimators': 50,
    
    'n_folds': 3,
    
    'seq_length': 270,
    'val_start': pd.to_datetime('01101989', format='%d%m%Y'),
    'val_end': pd.to_datetime('30091999', format='%d%m%Y')
}    

###############
# Prepare run #
###############


def get_args() -> Dict:
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=["train", "evaluate", "eval_robustness"])
    parser.add_argument('--camels_root', type=str, help="Root directory of CAMELS data set")
    parser.add_argument('--seed', type=int, required=False, help="Random seed")
    parser.add_argument('--run_dir', type=str, help="For evaluation mode. Path to run directory.")
    parser.add_argument('--run_dir_base', type=str, default="runs", help="For training mode. Path to store run directories in.")
    parser.add_argument('--run_name', type=str, required=False, help="For training mode. Name of the run.")
    parser.add_argument('--num_workers',
                        type=int,
                        default=1,
                        help="Number of parallel threads for training")
    parser.add_argument('--no_static',
                        type=bool,
                        default=False,
                        help="If True, trains Random Forest without static features")
    parser.add_argument('--model_dir', type=str, required=False, help="For training mode. If provided, uses parameters from this run, else searches for suitable parameters.")
    parser.add_argument('--train_start', type=str, help="Training start date (ddmmyyyy).")
    parser.add_argument('--train_end', type=str, help="Training end date (ddmmyyyy).")
    parser.add_argument('--basins', 
                        nargs='+', default=get_basin_list(),
                        help='List of basins')
    cfg = vars(parser.parse_args())
    
    cfg["train_start"] = pd.to_datetime(cfg["train_start"], format='%d%m%Y')
    cfg["train_end"] = pd.to_datetime(cfg["train_end"], format='%d%m%Y')

    # Validation checks
    if (cfg["mode"] == "train") and (cfg["seed"] is None):
        # generate random seed for this run
        cfg["seed"] = int(np.random.uniform(low=0, high=1e6))

    if (cfg["mode"] in ["evaluate", "eval_robustness"]) and (cfg["run_dir"] is None):
        raise ValueError("In evaluation mode a run directory (--run_dir) has to be specified")
        
    # combine global settings with user config
    cfg.update(GLOBAL_SETTINGS)

    if cfg["mode"] == "train":
        # print config to terminal
        for key, val in cfg.items():
            print(f"{key}: {val}")

    # convert path to PosixPath object
    cfg["camels_root"] = Path(cfg["camels_root"])
    if cfg["run_dir"] is not None:
        cfg["run_dir"] = Path(cfg["run_dir"])
    if cfg["run_dir_base"] is not None:
        cfg["run_dir_base"] = Path(cfg["run_dir_base"])
    if cfg["model_dir"] is not None:
        cfg["model_dir"] = Path(cfg["model_dir"])
    return cfg


def _setup_run(cfg: Dict) -> Dict:
    """Create folder structure for this run

    Parameters
    ----------
    cfg : dict
        Dictionary containing the run config

    Returns
    -------
    dict
        Dictionary containing the updated run config
    """
    if cfg["run_name"] is None:
        now = datetime.now()
        day = f"{now.day}".zfill(2)
        month = f"{now.month}".zfill(2)
        hour = f"{now.hour}".zfill(2)
        minute = f"{now.minute}".zfill(2)
        run_name = f'run_rf_{day}{month}_{hour}{minute}_seed{cfg["seed"]}'
    else:
        run_name = f'run_{cfg["run_name"]}_seed{cfg["seed"]}'
    cfg['run_dir'] = Path(__file__).absolute().parent / cfg["run_dir_base"] / run_name
    if not cfg["run_dir"].is_dir():
        cfg["train_dir"] = cfg["run_dir"] / 'data' / 'train'
        cfg["train_dir"].mkdir(parents=True)
        cfg["val_dir"] = cfg["run_dir"] / 'data' / 'val'
        cfg["val_dir"].mkdir(parents=True)
    else:
        raise RuntimeError('There is already a folder at {}'.format(cfg["run_dir"]))

    # dump a copy of cfg to run directory
    with (cfg["run_dir"] / 'cfg.json').open('w') as fp:
        temp_cfg = {}
        for key, val in cfg.items():
            if isinstance(val, PosixPath):
                temp_cfg[key] = str(val)
            elif isinstance(val, pd.Timestamp):
                temp_cfg[key] = val.strftime(format="%d%m%Y")
            elif 'param_dist' in key:
                temp_dict = {}
                for k, v in val.items():
                    if isinstance(v, sp.stats._distn_infrastructure.rv_frozen):
                        temp_dict[k] = f"{v.dist.name}{v.args}, *kwds={v.kwds}"
                    else:
                        temp_dict[k] = str(v)
                temp_cfg[key] = str(temp_dict)
            else:
                temp_cfg[key] = val
        json.dump(temp_cfg, fp, sort_keys=True, indent=4)

    return cfg


def _prepare_data(cfg: Dict, basins: List) -> Dict:
    """Preprocess training data.

    Parameters
    ----------
    cfg : dict
        Dictionary containing the run config
    basins : List
        List containing the 8-digit USGS gauge id

    Returns
    -------
    dict
        Dictionary containing the updated run config
    """
    # create database file containing the static basin attributes
    cfg["db_path"] = str(cfg["run_dir"] / "attributes.db")
    add_camels_attributes(cfg["camels_root"], db_path=cfg["db_path"])

    # create .h5 files for train and validation data
    cfg["train_file"] = cfg["train_dir"] / 'train_data.h5'
    create_h5_files(camels_root=cfg["camels_root"],
                    out_file=cfg["train_file"],
                    basins=basins,
                    dates=[cfg["train_start"], cfg["train_end"]],
                    with_basin_str=True,
                    seq_length=cfg["seq_length"])

    return cfg


###########################
# Train or evaluate model #
###########################


def train(cfg):
    """Train model.

    Parameters
    ----------
    cfg : Dict
        Dictionary containing the run config
    """
    # fix random seeds
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.cuda.manual_seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    basins = cfg["basins"]

    # create folder structure for this run
    cfg = _setup_run(cfg)

    # prepare data for training
    cfg = _prepare_data(cfg=cfg, basins=basins)

    # prepare Dataset
    ds = CamelsH5(h5_file=cfg["train_file"],
                  basins=basins,
                  db_path=cfg["db_path"],
                  concat_static=False,
                  cache=True,
                  no_static=cfg["no_static"])

    # Create train/val sets
    x = ds.x.reshape(len(ds.x), -1)
    y = ds.y.reshape(len(ds.y))
    if not cfg["no_static"]:
        attr_indices = np.searchsorted(ds.df.index, ds.sample_2_basin)
        attributes = ds.df.iloc[attr_indices].values
        x = np.concatenate([x, attributes], axis=1)
    
    print(x.shape, y.shape)
    if cfg["model_dir"] is None:
        # Find optimal number of iterations
        
        model = RandomForestRegressor(n_estimators=cfg["n_estimators"], n_jobs=cfg["num_workers"], random_state=cfg["seed"], verbose=10)
        model = model_selection.RandomizedSearchCV(model, cfg["param_dist"], n_iter=cfg["n_iter"], cv=cfg["n_folds"], return_train_score=True, 
                                                   scoring='neg_mean_squared_error', n_jobs=1, refit=True, random_state=cfg["seed"], verbose=10)
        model.fit(x, y)
        best_params = model.best_params_
        print(f"Best parameters: {best_params}")

        cv_results = pd.DataFrame(model.cv_results_).sort_values(by='mean_test_score', ascending=False)
        print(cv_results.filter(regex='param_|mean_test_score|mean_train_score', axis=1).head())
        print(cv_results.loc[model.best_index_, ['mean_train_score', 'mean_test_score']])
        
    else:
        print('Using model parameters from {}'.format(cfg["model_dir"]))
        model = pickle.load(open(cfg["model_dir"] / "model.pkl", "rb"))
        model_params = model.get_params()
        print(model_params)
        
        model = RandomForestRegressor(n_estimators=cfg["n_estimators"], n_jobs=cfg["num_workers"], random_state=cfg["seed"])
        model.set_params(**model_params)
        model.random_state = cfg["seed"]
        model.n_jobs = cfg["num_workers"]
        model.fit(x, y)
    
    model_path = cfg["run_dir"] / "model.pkl"
    pickle.dump(model, open(str(model_path), 'wb'))
    

def evaluate(user_cfg: Dict):
    """Train model for a single epoch.

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config
        
    """
    with open(user_cfg["run_dir"] / 'cfg.json', 'r') as fp:
        run_cfg = json.load(fp)

    basins = run_cfg["basins"]

    # get attribute means/stds
    db_path = str(user_cfg["run_dir"] / "attributes.db")
    attributes = load_attributes(db_path=db_path, 
                                 basins=basins,
                                 drop_lat_lon=True)
    means = attributes.mean()
    stds = attributes.std()
    
    # load trained model
    model_file = user_cfg["run_dir"] / 'model.pkl'
    model = pickle.load(open(model_file, 'rb'))

    date_range = pd.date_range(start=GLOBAL_SETTINGS["val_start"], end=GLOBAL_SETTINGS["val_end"])
    results = {}
    for basin in tqdm(basins):
        ds_test = CamelsTXT(camels_root=user_cfg["camels_root"],
                            basin=basin,
                            dates=[GLOBAL_SETTINGS["val_start"], GLOBAL_SETTINGS["val_end"]],
                            is_train=False,
                            seq_length=run_cfg["seq_length"],
                            with_attributes=True,
                            attribute_means=means,
                            attribute_stds=stds,
                            concat_static=False,
                            db_path=db_path)

        preds, obs = evaluate_basin(model, ds_test, run_cfg["no_static"])

        df = pd.DataFrame(data={'qobs': obs.flatten(), 'qsim': preds.flatten()}, index=date_range)

        results[basin] = df

    _store_results(user_cfg, run_cfg, results)


def evaluate_basin(model: RandomForestRegressor, ds_test: CamelsTXT, no_static: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate model on a single basin

    Parameters
    ----------
    model : RandomForestRegressor
        The RF model to evaluate
    ds_test : CamelsTXT
        CAMELS dataset containing the basin data
    no_static: bool
        If True, will not include static attributes as input features

    Returns
    -------
    preds : np.ndarray
        Array containing the (rescaled) prediction for the entire data period
    obs : np.ndarray
        Array containing the observed discharge for the entire data period

    """
    preds, obs = None, None

    x = ds_test.x.reshape(len(ds_test.x), -1).numpy()
    obs = ds_test.y.numpy()
    if not no_static:
        x = np.concatenate([x, ds_test.attributes.repeat(len(x), 1).numpy()], axis=1)
    
    preds = model.predict(x)
    preds = rescale_features(preds, variable='output')
    
    # set discharges < 0 to zero
    preds[preds < 0] = 0

    return preds, obs




def _store_results(user_cfg: Dict, run_cfg: Dict, results: pd.DataFrame):
    """Store results in a pickle file.

    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config
    run_cfg : Dict
        Dictionary containing the run config loaded from the cfg.json file
    results : pd.DataFrame
        DataFrame containing the observed and predicted discharge.

    """
    if run_cfg["no_static"]:
        file_name = user_cfg["run_dir"] / f"rf_no_static_seed{run_cfg['seed']}.p"
    else:
        file_name = user_cfg["run_dir"] / f"rf_seed{run_cfg['seed']}.p"

    with (file_name).open('wb') as fp:
        pickle.dump(results, fp)

    print(f"Sucessfully stored results at {file_name}")


if __name__ == "__main__":
    config = get_args()
    globals()[config["mode"]](config)
