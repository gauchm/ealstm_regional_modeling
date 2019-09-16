"""
Code to train and evaluate XGBoost models.
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
import xgboost as xgb
from sklearn import model_selection
from tqdm import tqdm

from papercode.datasets import CamelsH5, CamelsTXT
from papercode.datautils import (add_camels_attributes, load_attributes,
                                 rescale_features)
from papercode.metrics import calc_nse
from papercode.nseloss import NSELoss
from papercode.utils import create_h5_files, get_basin_list

###########
# Globals #
###########

# fixed settings for all experiments
GLOBAL_SETTINGS = {
    # XGBoost parameters
    #'learning_rate': 0.1,
    'n_estimators': 100,
    #'colsample_bylevel': 0.9,
    #'colsample_bytree': 0.9,
    #'subsample': 0.8,
    #'gamma': 3,
    #'max_depth': 5,
    #'min_child_weight': 4,
    #'reg_alpha': 20,
    #'reg_lambda': 0.5,
    'early_stopping_rounds': 40,
    'objective': 'reg:squarederror',
    
    # parameters for RandomSearchCV:
    'param_dist': {
        'learning_rate': [0.25],
        'gamma': sp.stats.uniform(0, 5),
        'max_depth': sp.stats.randint(2, 8),
        'min_child_weight': sp.stats.randint(1, 15),
        'subsample': [0.5],
        'colsample_bytree': sp.stats.uniform(0.3, 0.7),
        'colsample_bylevel': sp.stats.uniform(0.3, 0.7),
        #'reg_alpha': sp.stats.expon(0, 20),
        #'reg_lambda': sp.stats.expon(0, 20),
    },
    'n_iter': 10000,
    'n_cv': 2,
    
    'seq_length': 8,
    'train_start': pd.to_datetime('01101999', format='%d%m%Y'),
    'train_end': pd.to_datetime('30092008', format='%d%m%Y'),
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
    parser.add_argument('--num_workers',
                        type=int,
                        default=1,
                        help="Number of parallel threads for training")
    parser.add_argument('--no_static',
                        type=bool,
                        default=False,
                        help="If True, trains XGBoost without static features")
    parser.add_argument('--use_mse',
                        type=bool,
                        default=True,
                        help="If True, uses mean squared error as XGBoost objective.")
    cfg = vars(parser.parse_args())

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
    now = datetime.now()
    day = f"{now.day}".zfill(2)
    month = f"{now.month}".zfill(2)
    hour = f"{now.hour}".zfill(2)
    minute = f"{now.minute}".zfill(2)
    run_name = f'run_xgb_{day}{month}_{hour}{minute}_seed{cfg["seed"]}'
    cfg['run_dir'] = Path(__file__).absolute().parent / "runs" / run_name
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
            elif key == 'param_dist':
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

    basins = get_basin_list()

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
    
    # define loss function
    if not cfg["use_mse"]:
        raise NotImplementedError('NSE for XGBoost is not implemented')

    # Create train/val sets
    x = ds.x.reshape(len(ds.x), -1)
    y = ds.y.reshape(len(ds.y))
    if not cfg["no_static"]:
        attr_indices = np.searchsorted(ds.df.index, ds.sample_2_basin)
        attributes = ds.df.iloc[attr_indices].values
        x = np.concatenate([x, attributes], axis=1)
    
    num_val_samples = int(len(x) * 0.1)
    val_indices = np.random.choice(range(len(x)), size=num_val_samples, replace=False)
    train_indices = np.setdiff1d(range(len(x)), val_indices)

    val = [(x[train_indices], y[train_indices]), 
           (x[val_indices], y[val_indices])]

    param_search = len(cfg['param_dist'].keys()) > 0
    if param_search:
        if (cfg["n_iter"] * cfg["n_cv"] >= cfg["num_workers"]) or (cfg["num_workers"] == -1):
            n_jobs_xgb = 1 
            n_jobs_cv = cfg["num_workers"]
        else:
            n_jobs_xgb = 2
            n_jobs_cv = cfg["num_workers"] // 2
        model = xgb.XGBRegressor(n_estimators=cfg["n_estimators"], objective=cfg["objective"], n_jobs=n_jobs_xgb, random_state=cfg["seed"])
        model = model_selection.RandomizedSearchCV(model, cfg["param_dist"], n_iter=cfg["n_iter"], cv=cfg["n_cv"], return_train_score=True, 
                                                   scoring='neg_mean_squared_error', n_jobs=n_jobs_cv, random_state=cfg["seed"], verbose=5)
    else:
        model = xgb.XGBRegressor(n_estimators=cfg["n_estimators"], learning_rate=cfg["learning_rate"], reg_alpha=cfg["reg_alpha"], reg_lambda=cfg["reg_lambda"],
                                 subsample=cfg["subsample"], colsample_bylevel=cfg["colsample_bylevel"], colsample_bytree=cfg["colsample_bytree"], 
                                 gamma=cfg["gamma"], max_depth=cfg["max_depth"], min_child_weight=cfg["min_child_weight"], 
                                 n_jobs=cfg["num_workers"], random_state=cfg["seed"])
        
    model.fit(x[train_indices], y[train_indices], eval_set=val, eval_metric='rmse', 
              early_stopping_rounds=cfg["early_stopping_rounds"], verbose=not param_search)
    
    if param_search:
        cv_results = pd.DataFrame(model.cv_results_).sort_values(by='mean_test_score', ascending=False)
        print(cv_results.filter(regex='param_|mean_test_score|mean_train_score', axis=1).head())
        print('Best params: {}'.format(model.best_params_))
        print(cv_results.loc[model.best_index_, ['mean_train_score', 'mean_test_score']])

    
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

    basins = get_basin_list()

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


def evaluate_basin(model: xgb.XGBRegressor, ds_test: CamelsTXT, no_static: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate model on a single basin

    Parameters
    ----------
    model : xgb.XGBRegressor
        The XGBoost model to evaluate
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
    
    # set discharges < 0 to zero
    preds[preds < 0] = 0

    return preds, obs


def eval_robustness(user_cfg: Dict):
    """Evaluate model robustness of XGBoost

    In this experiment, gaussian noise with increasing scale is added to the static features to
    evaluate the model robustness against pertubations of the static catchment characteristics.
    For each scale, 50 noise vectors are drawn.
    
    Parameters
    ----------
    user_cfg : Dict
        Dictionary containing the user entered evaluation config
    
    Raises
    ------
    NotImplementedError
        If the run_dir specified points not to a XGBoost model folder.
    """
    random.seed(user_cfg["seed"])
    np.random.seed(user_cfg["seed"])

    # fixed settings for this analysis
    n_repetitions = 50
    scales = [0.1 * i for i in range(11)]

    with open(user_cfg["run_dir"] / 'cfg.json', 'r') as fp:
        run_cfg = json.load(fp)

    if run_cfg["no_static"]:
        raise NotImplementedError("This function only works with static attributes")

    basins = get_basin_list()

    # get attribute means/stds
    db_path = str(user_cfg["run_dir"] / "attributes.db")
    attributes = load_attributes(db_path=db_path, 
                                 basins=basins,
                                 drop_lat_lon=True)
    means = attributes.mean()
    stds = attributes.std()

    # initialize Model
    model_file = user_cfg["run_dir"] / "model.pkl"
    model = pickle.load(open(model_file, 'rb'))

    overall_results = {}
    # process bar handle
    pbar = tqdm(basins, file=sys.stdout)
    for basin in pbar:
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
        
        basin_results = defaultdict(list)
        step = 1
        for scale in scales:
            for _ in range(1 if scale == 0.0 else n_repetitions):
                noise = np.random.normal(loc=0, scale=scale, size=27).astype(np.float32)
                noise = torch.from_numpy(noise)
                nse = eval_with_added_noise(model, ds_test, noise)
                basin_results[scale].append(nse)
                pbar.set_postfix_str(f"Basin progress: {step}/{(len(scales)-1)*n_repetitions+1}")
                step += 1

        overall_results[basin] = basin_results
    out_file = (Path(__file__).absolute().parent /
                'results/{}_model_robustness.p'.format(user_cfg["run_dir"].name))
    if not out_file.parent.is_dir():
        out_file.parent.mkdir(parents=True)
    with out_file.open("wb") as fp:
        pickle.dump(overall_results, fp)


def eval_with_added_noise(model: xgb.XGBRegressor, ds_test: CamelsTXT, noise: torch.Tensor) -> float:
    """Evaluate model on a single basin with added noise

    Parameters
    ----------
    model : xgb.XGBRegressor
        The XGBoost model to evaluate
    ds_test : CamelsTXT
        Dataset containing the basin data.
    noise : torch.Tensor
        Tensor containing the noise for this evaluation run.
    
    Returns
    -------
    float
        Nash-Sutcliffe-Efficiency of the simulations with added noise.
    """
    preds, obs = None, None
                
    x = ds_test.x.reshape(len(ds_test.x), -1).numpy()
    obs = ds_test.y.numpy()
    
    attributes = ds_test.attributes.repeat(len(x), 1).clone()
    batch_noise = noise.repeat(len(attributes), 1)
    attributes = attributes.add(batch_noise)
    x = np.concatenate([x, attributes.numpy()], axis=1)
    preds = model.predict(x)

    preds = rescale_features(preds, variable='output')

    # set discharges < 0 to zero
    preds[preds < 0] = 0

    nse = calc_nse(obs[obs >= 0], preds[obs.reshape(-1) >= 0])
    return nse


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
        file_name = user_cfg["run_dir"] / f"xgboost_no_static_seed{run_cfg['seed']}.p"
    else:
        file_name = user_cfg["run_dir"] / f"xgboost_seed{run_cfg['seed']}.p"

    with (file_name).open('wb') as fp:
        pickle.dump(results, fp)

    print(f"Sucessfully stored results at {file_name}")


if __name__ == "__main__":
    config = get_args()
    globals()[config["mode"]](config)
