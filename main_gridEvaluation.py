"""
Code to create SLURM job files to compare XGBoost and LSTM models on different amounts of training data.
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
import pandas as pd

from papercode.utils import get_basin_list

###########
# Globals #
###########

# fixed settings for all experiments
GLOBAL_SETTINGS = {
    # parameters for RandomSearchCVs:
    'train_ranges': {9: ('01101999', '30092008'), 6: ('01101999', '30092005'), 3: ('01101999', '30092002')},
    'n_basins': [531, 265, 53],
    'basin_samples_per_grid_cell': 5,
    'ealstm_time': 0.3,  # minutes per year and basin
    'ealstm_memory': 0.09,  # G per basin
    'xgb_time': 0.3,  # minutes per year and basin
    'xgb_time_paramsearch': 4.1, # minutes per basin
    'xgb_memory': 0.2354,  # G per basin
    
    'seeds': [111, 222, 333, 444, 555, 666, 777, 888],
    
    'seq_length': 270,
    'val_start': pd.to_datetime('01101989', format='%d%m%Y'),
    'val_end': pd.to_datetime('30091999', format='%d%m%Y')
}

ealstm_sbatch_template = \
"""
#!/bin/bash
#SBATCH --account=def-kshook
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task={num_workers}
#SBATCH --mem={memory}       # memory (per node)
#SBATCH --time={time}        # time (DD-HH:MM)
source /home/mgauch/.bashrc
conda activate ealstm

date
python main.py train --camels_root {camels_root} --seed {seed} --cache_data True --basins {basins} --num_workers {num_workers} --train_start {train_start} --train_end {train_end}
date
"""

xgb_sbatch_template = \
"""
#!/bin/bash
#SBATCH --account=def-kshook
#SBATCH --cpus-per-task={num_workers}
#SBATCH --mem={memory}       # memory (per node)
#SBATCH --time={time}        # time (DD-HH:MM)
source /home/mgauch/.bashrc
conda activate ealstm

date
python main_xgboost_limitedData.py train --camels_root {camels_root} --seed {seed} --basins {basins} --num_workers {num_workers} --train_start {train_start} --train_end {train_end} --run_dir_base {run_dir_base} {options}
date
"""

def get_args() -> Dict:
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--camels_root', type=str, help="Root directory of CAMELS data set")
    parser.add_argument('--num_workers_ealstm', type=int, default=12, help="Number of parallel threads for EALSTM training")
    parser.add_argument('--num_workers_xgb', type=int, default=20, help="Number of parallel threads for XGBoost training")
    
    cfg = vars(parser.parse_args())
    cfg.update(GLOBAL_SETTINGS)
    
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
    cfg["run_name"] = f'run_grid_{day}{month}_{hour}{minute}'
    cfg['run_dir'] = Path(__file__).absolute().parent / "runs" / cfg["run_name"]
    if cfg["run_dir"].is_dir():
        raise RuntimeError('There is already a folder at {}'.format(cfg["run_dir"]))
    else:
        cfg["run_dir"].mkdir(parents=True)

    # dump a copy of cfg to run directory
    with (cfg["run_dir"] / 'cfg.json').open('w') as fp:
        temp_cfg = {}
        for key, val in cfg.items():
            if isinstance(val, PosixPath):
                temp_cfg[key] = str(val)
            elif isinstance(val, pd.Timestamp):
                temp_cfg[key] = val.strftime(format="%d%m%Y")
            else:
                temp_cfg[key] = val
        json.dump(temp_cfg, fp, sort_keys=True, indent=4)

    return cfg


################
# Prepare grid #
################
if __name__ == "__main__":
    cfg = get_args()
    cfg = _setup_run(cfg)
    
    np.random.seed(0)
    basins = get_basin_list()
    basin_samples = []
    for n_basins in cfg["n_basins"]:
        for i in range(cfg["basin_samples_per_grid_cell"]):
            basin_samples.append(np.random.choice(basins, size=n_basins, replace=False))
            if n_basins == 531:
                break
                
    for n_years, train_range in cfg["train_ranges"].items():
        train_start, train_end = train_range
        for i, basin_sample in enumerate(basin_samples):
            ealstm_time = int(cfg["ealstm_time"] * len(basin_sample) * n_years)
            ealstm_mem = int(cfg["ealstm_memory"] * len(basin_sample))
            xgb_time = int(cfg["xgb_time"] * len(basin_sample) * n_years)
            xgb_time_paramsearch = int(cfg["xgb_time_paramsearch"] * len(basin_sample))
            xgb_mem = int(cfg["xgb_memory"] * len(basin_sample))
            
            # Do the XGB parameter search for one seed, then reuse these parameters for all seeds
            param_search_name = f"xgb_param_search_{train_start}_{train_end}_basinsample{len(basin_sample)}_{i}"
            param_search_model_dir = cfg["run_dir"] / param_search_name
            xgb_train_str = xgb_sbatch_template.format(basins=' '.join(basin_sample), seed=cfg["seeds"][0],
                                                       train_start=train_start, train_end=train_end, options=f"--run_name {param_search_name}",
                                                       time=f"00-00:{xgb_time_paramsearch}", memory=f"00-00:{xgb_mem}", camels_root=cfg["camels_root"], 
                                                       num_workers=cfg["num_workers_xgb"], run_dir_base=cfg["run_name"])
            
            with open(cfg["run_dir"] / f"{param_search_model_dir}.sbatch", "w") as f:
                f.write(xgb_train_str)
            
            # Training
            for seed in cfg["seeds"]:
                if seed != cfg["seeds"][0]:
                    with open(cfg["run_dir"] / f"xgb_train_{train_start}_{train_end}_basinsample{len(basin_sample)}_{i}_seed{seed}.sbatch", "w") as f:
                        xgb_train_str = xgb_sbatch_template.format(basins=' '.join(basin_sample), seed=seed, train_start=train_start, train_end=train_end, 
                                                                   options=f"--model_dir {param_search_model_dir}", time=f"00-00:{xgb_time}", 
                                                                   memory=f"00-00:{xgb_mem}", camels_root=cfg["camels_root"], num_workers=cfg["num_workers_xgb"], 
                                                                   run_dir_base=cfg["run_name"])
                        f.write(xgb_train_str)
                with open(cfg["run_dir"] / f"ealstm_train_{train_start}_{train_end}_basinsample{len(basin_sample)}_{i}_seed{seed}.sbatch", "w") as f:
                    ealstm_train_str = ealstm_sbatch_template.format(basins=' '.join(basin_sample), seed=seed, train_start=train_start, train_end=train_end,
                                                                     time=f"00-00:{ealstm_time}", memory=f"{ealstm_mem}G", camels_root=cfg["camels_root"], 
                                                                     num_workers=cfg["num_workers_ealstm"], run_dir_base=cfg["run_name"])
                    f.write(ealstm_train_str)