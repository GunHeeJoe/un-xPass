from pathlib import Path
import pandas as pd
from functools import partial
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow.utils.autologging_utils")

import torch
import os, sys
sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())))

from unxpass.databases import SQLiteDatabase
from unxpass.datasets import PassesDataset
from unxpass.components import pass_selection, pass_value, pass_success
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

STORES_FP = Path("../stores")

dataset_train = partial(PassesDataset, path=STORES_FP / "datasets" / "default" / "train")
dataset_test = partial(PassesDataset, path=STORES_FP / "datasets" / "default" / "test")

Logistic_model = MLPClassifier(hidden_layer_sizes=(),solver='adam',
                            beta_1=0.9, beta_2=0.999, early_stopping=True,max_iter=1000,
                            validation_fraction=0.25)

from unxpass import features as fs
all_feature_dict = [f.__name__ for f in fs.all_features]

test_feature_dict = {
    'speed' : [
        'speedx_a01',
        'speedy_a01',
        'speed_a01',
        'speedx_a02',
        'speedy_a02',
        'speed_a02',
    ]
}
small_feature_dict = {
        'startlocation' : [
            'start_x_a0',
            'start_y_a0'
        ],
        'endlocation' : [
            'end_x_a0',
            'end_y_a0'
        ],
        'angle' : [
            'angle_a0'
        ],
        'movement': [
            'movement_a0',
        ]
}
many_feature_dict = {
        'startlocation' : [
            'start_x_a0',
            'start_y_a0'
        ],
        'endlocation' : [
            'end_x_a0',
            'end_y_a0'
        ],
        'angle' : [
            'angle_a0'
        ],
        'startpolar': [
            'start_dist_to_goal_a0',
            'start_angle_to_goal_a0'
        ],
        'relative_startlocation': [
            'start_dist_goalline_a0',
            'start_dist_sideline_a0'
        ],
        'endpolar': [
            'end_dist_to_goal_a0',
            'end_angle_to_goal_a0'
        ],
        'relative_endlocation': [
            'end_dist_goalline_a0',
            'end_dist_sideline_a0'
        ],
        'movement': [
            'movement_a0',
            'dx_a0',
            'dy_a0'
        ],
        'ball_height_onehot': [
            'ball_height_ground_a0',
            'ball_height_low_a0',
            'ball_height_high_a0'
        ],
        'player_possession_time': [
            'player_possession_time_a0'
        ],
        'speed': [
            'speed_a01',
            'speed_a02'
        ],
        'under_pressure': [
            'under_pressure_a0'
        ],
        'dist_defender': [
            'dist_defender_start_a0',
            'dist_defender_end_a0',
            'dist_defender_action_a0'
        ],
        'nb_opp_in_path': [
            'nb_opp_in_path_a0'
        ]
}

#LogisticNet
LogisticNet = pass_success.SklearnComponent(
    model=Logistic_model,
    features=many_feature_dict
    
    #features=small_feature_dict
    #features=all_feature_dict
    #features= test_feature_dict
)

# LogisticNet.train(dataset_train)
# surface_pass_success = LogisticNet.test(dataset_test)
# print("Logistic pass success : ",surface_pass_success)

#Dense2Net
in_channel = len(many_feature_dict)
hidden_sizes = (in_channel)
#기본적으로 MLPClassifier의 출력층은 마지막에 sigmoid를 취하는것 같음
Dense2Net_model = MLPClassifier(hidden_layer_sizes=hidden_sizes, activation='relu',
                          solver='adam',beta_1=0.9, beta_2=0.999, early_stopping=True,
                          max_iter=1000,validation_fraction=0.25)

Dense2Net = pass_success.SklearnComponent(
    model=Dense2Net_model,
    features=many_feature_dict
    #features=small_feature_dict
)
# Dense2Net.train(dataset_train)
# surface_pass_success = Dense2Net.test(dataset_test)
# print("Dense2Net pass success : ",surface_pass_success," \n")

#XGBoost
parameters = {
    'num_round' : 100,
    'max_depth': 9,
    'eta': 0.25,
    'alpha': 1e-8,
    'lambda': 100,
    'gamma': 1e-8,
    'early_stopping_rounds': 100,
    'eval_metric': 'logloss',
    'objective': 'binary:logistic',
}
XGBoost_model = pass_success.XGBoostComponent(
    model=XGBClassifier(**parameters
        # you probably want to do some hyperparameter tuning here to get a good model
    ),
    #features=many_feature_dict
    #features=small_feature_dict
    features=all_feature_dict
)

XGBoost_model.train(dataset_train)
surface_pass_success = XGBoost_model.test(dataset_test)
print("XGBoost pass success : ",surface_pass_success)

df_imp = pd.DataFrame({'imp':XGBoost_model.model.feature_importances_}, index = XGBoost_model.model.get_booster().feature_names)
df_imp = df_imp.sort_values('imp').sort_values(by='imp',ascending=False).copy()

feat_num = df_imp.shape[0]
print("total number of features =", feat_num)
for index,value in zip(df_imp.index,df_imp['imp']):
    print(f"{index} : {value}")

# torch.save(LogisticNet, "./Pretrained_pass_success_model/LogisticNet")
# torch.save(Dense2Net,"./Pretrained_pass_success_model/Dense2Net")
# torch.save(XGBoost_model, "./Pretrained_pass_success_model/XGBoost")
