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
from papercode.nseloss import NSELoss, NSEObjective
from papercode.utils import create_h5_files, get_basin_list

###########
# Globals #
###########

# fixed settings for all experiments
GLOBAL_SETTINGS = {
    # parameters for RandomSearchCVs:
    'tree_param_dist': {
        'max_depth': sp.stats.randint(2, 8),
        'min_child_weight': sp.stats.randint(1, 15), 
    },
    'tree_n_iter': 100,
    
    'gamma_param_dist': { 'gamma': sp.stats.uniform(0, 5) },
    'gamma_n_iter': 10,
    
    'subsample_param_dist': {
        'subsample': sp.stats.uniform(0.3, 0.7),
        'colsample_bylevel': sp.stats.uniform(0.3, 0.7),
        'colsample_bytree': sp.stats.uniform(0.3, 0.7),
    },
    'subsample_n_iter': 800,
    
    'reg_param_dist': {
        'reg_alpha': sp.stats.expon(0, 20),
        'reg_lambda': sp.stats.expon(0, 20),
    },
    'reg_n_iter': 400,
    
    'lr_param_dist': { 'learning_rate': sp.stats.uniform(0.001, 0.3) },
    'lr_n_iter': 200,
    
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
                        help="If True, trains XGBoost without static features")
    parser.add_argument('--model_dir', type=str, required=False, help="For training mode. If provided, uses XGBoost parameters from this run, else searches for suitable parameters.")
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
        run_name = f'run_xgb_{day}{month}_{hour}{minute}_seed{cfg["seed"]}'
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
        
    # define loss function
    # slight hack to enable NSE on XGBoost: replace the target with a unique id
    # so we can figure out the corresponding q_std during the loss calculation.
    y_actual = y.copy()
    y = np.arange(len(y))
    loss = NSEObjective(y, y_actual, ds.q_stds)
    objective = loss.nse_objective
    objective_non_sklearn = loss.nse_objective_non_sklearn  # xgb.cv needs a different signature for custom objective functions
    eval_metric = loss.nse_metric
    scoring = loss.neg_nse_metric_sklearn
    
    num_val_samples = int(len(x) * 0.1)
    val_indices = np.random.choice(range(len(x)), size=num_val_samples, replace=False)
    train_indices = np.setdiff1d(range(len(x)), val_indices)

    val = [(x[train_indices], y[train_indices]), 
           (x[val_indices], y[val_indices])]

    if cfg["model_dir"] is None:
        # Find optimal number of iterations
        learning_rate = 0.25
        model = xgb.XGBRegressor(n_estimators=5000, learning_rate=learning_rate, max_depth=5, gamma=0.1, subsample=0.8, 
                                 colsample_bytree=0.8, n_jobs=cfg["num_workers"], random_state=cfg["seed"])
        xgb_param = model.get_xgb_params()
        xgb_train = xgb.DMatrix(x[train_indices], label=y[train_indices])
        cv_results = xgb.cv(xgb_param, xgb_train, num_boost_round=model.get_params()['n_estimators'], nfold=3, verbose_eval=True,
                           obj=objective_non_sklearn, feval=eval_metric, early_stopping_rounds=50, seed=cfg["seed"])
        best_n_estimators = cv_results.shape[0]
        print(cv_results.tail())
        print(f"Best n_estimators: {best_n_estimators}")

        # Search tree parameters
        model = xgb.XGBRegressor(n_estimators=best_n_estimators, learning_rate=learning_rate, gamma=0.1, subsample=0.8, 
                                 colsample_bylevel=0.8, objective=objective, n_jobs=1, random_state=cfg["seed"])
        model = model_selection.RandomizedSearchCV(model, cfg["tree_param_dist"], n_iter=cfg["tree_n_iter"], cv=3, return_train_score=True, 
                                                   scoring=scoring, n_jobs=cfg["num_workers"], refit=False, random_state=cfg["seed"], verbose=5)
        model.fit(x[train_indices], y[train_indices], eval_set=val, eval_metric=eval_metric, verbose=False)
        best_params = model.best_params_
        print(f"Best tree parameters: {best_params}")

        # Search for gamma
        for k,v in best_params.items():
            cfg["gamma_param_dist"][k] = [v]
        model = xgb.XGBRegressor(n_estimators=best_n_estimators, learning_rate=learning_rate, subsample=0.8, 
                                 colsample_bylevel=0.8, objective=objective, n_jobs=1, random_state=cfg["seed"])
        model = model_selection.RandomizedSearchCV(model, cfg["gamma_param_dist"], n_iter=cfg["gamma_n_iter"], cv=3, return_train_score=True, 
                                                   scoring=scoring, n_jobs=cfg["num_workers"], refit=False, random_state=cfg["seed"], verbose=5)
        model.fit(x[train_indices], y[train_indices], eval_set=val, eval_metric=eval_metric, verbose=False)
        best_params = model.best_params_
        print(f"Best gamma parameters: {best_params}")

        # Search subsample parameters
        for k,v in best_params.items():
            cfg["subsample_param_dist"][k] = [v]
        model = xgb.XGBRegressor(n_estimators=best_n_estimators, learning_rate=learning_rate, objective=objective, n_jobs=1, random_state=cfg["seed"])
        model = model_selection.RandomizedSearchCV(model, cfg["subsample_param_dist"], n_iter=cfg["subsample_n_iter"], cv=3, return_train_score=True, 
                                                   scoring=scoring, n_jobs=cfg["num_workers"], refit=False, random_state=cfg["seed"], verbose=5)
        model.fit(x[train_indices], y[train_indices], eval_set=val, eval_metric=eval_metric, verbose=False)
        best_params = model.best_params_
        print(f"Best subsample parameters: {best_params}")

        # Search regularization parameters
        for k,v in best_params.items():
            cfg["reg_param_dist"][k] = [v]
        model = xgb.XGBRegressor(n_estimators=best_n_estimators, learning_rate=learning_rate, objective=objective, n_jobs=1, random_state=cfg["seed"])
        model = model_selection.RandomizedSearchCV(model, cfg["reg_param_dist"], n_iter=cfg["reg_n_iter"], cv=3, return_train_score=True, 
                                                   scoring=scoring, n_jobs=cfg["num_workers"], refit=False, random_state=cfg["seed"], verbose=5)
        model.fit(x[train_indices], y[train_indices], eval_set=val, eval_metric=eval_metric, verbose=False)
        best_params = model.best_params_
        print(f"Best regularization parameters: {best_params}")

        cv_results = pd.DataFrame(model.cv_results_).sort_values(by='mean_test_score', ascending=False)
        print(cv_results.filter(regex='param_|mean_test_score|mean_train_score', axis=1).head())
        print(cv_results.loc[model.best_index_, ['mean_train_score', 'mean_test_score']])

        # Search learning rate
        best_lr = None, None, np.inf  # lr, n_estimators, score
        for lr in cfg["lr_param_dist"]["learning_rate"].rvs(size=cfg["lr_n_iter"]):
            model = xgb.XGBRegressor(n_estimators=100000, learning_rate=lr, n_jobs=cfg["num_workers"], random_state=cfg["seed"])
            xgb_param = model.get_xgb_params()
            for k,v in best_params.items():
                xgb_param[k] = v
            xgb_train = xgb.DMatrix(x[train_indices], label=y[train_indices])
            cv_results = xgb.cv(xgb_param, xgb_train, num_boost_round=model.get_params()['n_estimators'], nfold=3, verbose_eval=True,
                               obj=objective_non_sklearn, feval=eval_metric, early_stopping_rounds=500, seed=cfg["seed"])
            print(f"Best n_estimators for lr {lr}: {cv_results.shape[0]}")
            mean_score = cv_results.iloc[-1]['test-nse-mean']
            if mean_score < best_lr[2]:
                best_lr = lr, cv_results.shape[0], mean_score
        print(f"Best lr: {best_lr[0]} with mean score {best_lr[2]} and best n_estimators {best_lr[1]}")

        xgb_param["n_estimators"] = best_lr[1]
        xgb_param["learning_rate"] = best_lr[0]
        print(f"Final parameters: {xgb_param}")
        
    else:
        print('Using model parameters from {}'.format(cfg["model_dir"]))
        model = pickle.load(open(cfg["model_dir"] / "model.pkl", "rb"))
        xgb_param = model.get_xgb_params()
        print(xgb_param)
        
    model = xgb.XGBRegressor()
    model.set_params(**xgb_param)
    model.objective = objective
    model.random_state = cfg["seed"]
    model.n_jobs = cfg["num_workers"]
    model.fit(x[train_indices], y[train_indices], eval_set=val, eval_metric=eval_metric, verbose=True)    
    
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
        file_name = user_cfg["run_dir"] / f"xgboost_no_static_seed{run_cfg['seed']}.p"
    else:
        file_name = user_cfg["run_dir"] / f"xgboost_seed{run_cfg['seed']}.p"

    with (file_name).open('wb') as fp:
        pickle.dump(results, fp)

    print(f"Sucessfully stored results at {file_name}")


if __name__ == "__main__":
    config = get_args()
    globals()[config["mode"]](config)
