from pathlib import Path
from functools import partial

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import mlflow
from scipy.ndimage import zoom
import pickle
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow.utils.autologging_utils")

import torch
import os, sys
import tqdm
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))
from unxpass.components.utils import save_model, load_model, log_model
from unxpass.databases import SQLiteDatabase
from unxpass.datasets import PassesDataset, CompletedPassesDataset, FailedPassesDataset
from unxpass.components import pass_selection, pass_value, pass_success
from unxpass.sampler import My_Sampler

import json
plt_settings = {"cmap": "magma", "vmin": 0, "vmax": 1, "interpolation": "bilinear"}

STORES_FP = Path("../stores")
PATH = './Pretrained_model/'

dataset_train = partial(PassesDataset, path=STORES_FP / "datasets" / "default" / "train")
dataset_test = partial(PassesDataset, path=STORES_FP / "datasets" / "default" / "test")

# Completed_dataset_train = partial(CompletedPassesDataset, path=STORES_FP / "datasets" / "default" / "train")
# Completed_dataset_test = partial(CompletedPassesDataset, path=STORES_FP / "datasets" / "default" / "test")
# Failed_dataset_train = partial(FailedPassesDataset, path=STORES_FP / "datasets" / "default" / "train")
# Failed_dataset_test = partial(FailedPassesDataset, path=STORES_FP / "datasets" / "default" / "test")

def pass_surface_save(path, surface):
    with open(path, 'wb') as f:
        pickle.dump(surface, f)

# print("pass selection modeling")
# model_pass_selection = pass_selection.SoccerMapComponent(
#     pass_selection.PytorchSoccerMapModel() )

# model_pass_selection.train(dataset_train)
# metric_pass_selection = model_pass_selection.test(dataset_test)
# print("pass selection metric : ",metric_pass_selection)

#surface_selection = model_pass_selection.predict_surface(dataset_test)
# pass_surface_save('./pass-surface/Intended_pass_selection.pkl', surface_selection)

print("pass success modeling")
model_pass_success = pass_success.SoccerMapComponent(
    pass_success.PytorchSoccerMapModel()) 

model_pass_success.train(dataset_train)
metric_pass_success = model_pass_success.test(dataset_test)
print("pass success metric: ",metric_pass_success)

surface_pass_success = model_pass_success.predict_surface(dataset_test)
pass_surface_save('./pass-surface/End_Intended_pass_Success.pkl', surface_pass_success)

# model_pass_successful_value = pass_value.SoccerMapComponent(
#     pass_value.PytorchSoccerMapModel() )

# model_pass_successful_value.train(dataset_train)
# surface_pass_value = model_pass_successful_value.test(dataset_test)
# print("completed pass value : ",surface_pass_value)

# model_pass_missed_value = pass_value.SoccerMapComponent(
#     pass_value.PytorchSoccerMapModel(),offensive=False, success=False)

# model_pass_missed_value.train(dataset_train)
# surface_pass_value = model_pass_missed_value.test(dataset_test)
# print("Failed pass value : ",surface_pass_value)


