"""Implements the pass selection component."""
import json
import math
from typing import Any, Dict, List, Optional

import hydra
import mlflow
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import xgboost as xgb
from rich.progress import track
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from unxpass.components.soccermap import SoccerMap, pixel
from unxpass.config import logger as log
from .soccermap import SoccerMap, pixel, soccermap_ToSoccerMapTensor
from .base import UnxpassComponent, UnxPassPytorchComponent, UnxPassXGBoostComponent
from torchmetrics.classification import BinaryCalibrationError
from sklearn.preprocessing import StandardScaler

class PassSelectionComponent(UnxpassComponent):
    """The pass selection component.

    From any given game situation where a player controls the ball, the model
    estimates the most likely destination of a potential pass.
    """

    component_name = "pass_selection"

    def _get_metrics(self, y, y_hat):

        return {
            "log_loss": log_loss(y, y_hat, labels=[0, 1]),
            "brier": brier_score_loss(y, y_hat),
            "ECE" : self.expected_calibration_error(y_hat,y),
        }


class ClosestPlayerBaseline(PassSelectionComponent):
    """A baseline model that predicts the closest player to the ball as the most likely receiver."""

    def __init__(self):
        super().__init__(
            features={
                "pass_options": ["distance"],
            },
            label=["receiver"],
            
        )

    def train(self, dataset, optimized_metric='neg_log_loss'):
        # No training required
        # Return metric score for hyperparameter optimization
        if optimized_metric is not None:
            metrics = self.test(dataset)
            return metrics[optimized_metric]

        return None

    def test(self, dataset) -> Dict:
        data = self.initialize_dataset(dataset)
        
        X_test,y = data.features, data.labels
    
        #공-모든 팀원간의 거리
        y["pred"] = X_test.distance
        y["pred_rank"] = (
            #공 - 모든 팀원간 거리기반으로 오름차순해서 가장 가까운 선수를 recevier라고 예측하는 모델
            y.groupby(["game_id", "action_id"])["pred"].rank("dense", ascending=True).astype(int)
        )
        return {"acc": len(y[y.receiver & (y.pred_rank == 1)]) / y.receiver.sum()}

    def predict(self, dataset) -> pd.Series:
        data = self.initialize_dataset(dataset)
        pred = data.features[["distance"]]
        pred["rank"] = (
            pred.groupby(["game_id", "action_id"])["distance"]
            .rank("dense", ascending=True)
            .astype(int)
        )
        return pd.Series(
            (pred["rank"] == 1).astype(float).values.tolist(),
            index=data.features.index,
        )


class XGBoostComponent(PassSelectionComponent, UnxPassXGBoostComponent):
    """A XGBoost model based on handcrafted features."""
        
    def __init__(
        self,
        model: xgb.XGBRanker,
        features: Dict[str, List[str]],
        label: List[str] = ["receiver"],
    ):
        super().__init__(
            model=model,
            features= {
                "startlocation": ["start_x_a0", "start_y_a0"],
                "endlocation": ["end_x_a0", "end_y_a0"],
                "speed": ["speedx_a02", "speedy_a02"],
                "freeze_frame_360": ["freeze_frame_360_a0"],
                "dist_defender" : ["dist_defender_start_a0", "dist_defender_end_a0", "dist_defender_action_a0"]
            },
            label=label
        )

    def train(self, dataset, optimized_metric=None, **train_cfg) -> Optional[float]:
        mlflow.xgboost.autolog()

        # Load data
        data = self.initialize_dataset(dataset)
        print(data.features)
        print(data.labels)
        X_train, X_val, y_train, y_val = train_test_split(
            data.features, data.labels, test_size=0.2
        )
        group_train = X_train.groupby(["game_id", "action_id"]).size()
        group_val = X_val.groupby(["game_id", "action_id"]).size()

        # Train the model
        log.info("Fitting model on train set")
        self.model.fit(
            X_train,
            y_train,
            group=group_train,
            eval_set=[(X_val, y_val)],
            eval_group=[group_val],
            **train_cfg,
        )

        # Return metric score for hyperparameter optimization
        if optimized_metric is not None:
            idx = self.model.best_iteration
            return self.model.evals_result()["validation_0"][optimized_metric][idx]

        return None

    def test(self, dataset) -> Dict[str, float]:
        data = self.initialize_dataset(dataset)
        X_test, y = data.features, data.labels
        y["pred"] = self.model.predict(X_test)
        y["pred_rank"] = (
            y.groupby(["game_id", "action_id"])["pred"].rank("dense", ascending=False).astype(int)
        )
        return {"acc": len(y[y.receiver & (y.pred_rank == 1)]) / y.receiver.sum()}


class PytorchSoccerMapModel(pl.LightningModule, UnxPassPytorchComponent):
    
    """A pass selection model based on the SoccerMap architecture."""

    def __init__(
        self,
        lr: float = 1e-5,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = SoccerMap(in_channels=9)
        self.softmax = nn.Softmax(2)

        # loss function
        self.criterion = torch.nn.BCELoss()

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        x = self.softmax(x.view(*x.size()[:2], -1)).view_as(x)
        
        return x

    def step(self, batch: Any):
        x, mask, y = batch
        surface = self.forward(x)
        #해당 목적지의 단일 예측값만 loss로 사용
        y_hat = pixel(surface, mask)
        loss = self.criterion(y_hat, y)
        return loss, y_hat, y

    #batch는 데이터셋으로 dataset_train을 불러옴
    def training_step(self, batch: Any, batch_idx: int):
        data = self.initialize_dataset(batch)

        dataloader = DataLoader(
            data,
            shuffle=False,
            batch_size=32,
            num_workers=0,
            pin_memory=False,
        )
        loss, preds, targets = self.step(batch)

        #loss, preds, targets = self.step(dataloader)

        # log train metrics
        self.log("loss/train", loss, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
       
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        self.log("loss/val", loss, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        #self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def predict_step(self, batch: Any, batch_idx: int):
        x, _, _ = batch
        surface = self(x)
        return surface
    
    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)


class ToSoccerMapTensor:
    """Convert inputs to a spatial representation.

    Parameters
    ----------
    dim : tuple(int), default=(68, 104)
        The dimensions of the pitch in the spatial representation.
        The original pitch dimensions are 105x68, but even numbers are easier
        to work with.
    """

    def __init__(self, dim=(68, 104)):
        assert len(dim) == 2
        self.y_bins, self.x_bins = dim

    def _get_cell_indexes(self, x, y):
        x_bin = np.clip(x / 105 * self.x_bins, 0, self.x_bins - 1).astype(np.uint8)
        y_bin = np.clip(y / 68 * self.y_bins, 0, self.y_bins - 1).astype(np.uint8)
        return x_bin, y_bin

    def __call__(self, sample, train_test):
        start_x, start_y, end_x, end_y = (
            sample["start_x_a0"],
            sample["start_y_a0"],
            sample["end_x_a0"],
            sample["end_y_a0"],
        )
   
        intended_end_x, intended_end_y = (
            sample["intended_end_x_a0"],
            sample["intended_end_y_a0"],
        )   
        ball_height_ground, ball_height_low, ball_height_high = (
            sample["ball_height_ground_a0"],
            sample["ball_height_low_a0"],
            sample["ball_height_high_a0"]
        )
        

        #intended_end_x, intended_end_y = (sample['intended_end_x_a0'], sample['intended_end_y_a0'])
        scaler = StandardScaler()
        speed_x, speed_y = sample["speedx_a02"], sample["speedy_a02"]
        frame = pd.DataFrame.from_records(sample["freeze_frame_360_a0"])
        target = int(sample["success"]) if "success" in sample else None

        # Location of the player that passes the ball
        # passer_coo = frame.loc[frame.actor, ["x", "y"]].fillna(1e-10).values.reshape(-1, 2)
        # Location of the ball
        ball_coo = np.array([[start_x, start_y]])
        ball_end_coo = np.array([[end_x, end_y]])
        # Location of the goal
        goal_coo = np.array([[105, 34]])

        # Output
        matrix = np.zeros((9, self.y_bins, self.x_bins))

        # Locations of the passing player's teammates
        players_att_coo = frame.loc[~frame.actor & frame.teammate, ["x", "y"]].values.reshape(-1, 2)
        # Locations and speed vector of the defending players
        players_def_coo = frame.loc[~frame.teammate, ["x", "y"]].values.reshape(-1, 2)

        # 현재 상태의 위치정보
        # CH 1: Locations of attacking team
        # 공격팀의 위치정보
        x_bin_att, y_bin_att = self._get_cell_indexes(
            players_att_coo[:, 0],
            players_att_coo[:, 1],
        )
        matrix[0, y_bin_att, x_bin_att] = 1

        # CH 2: Locations of defending team
        # 수비팀의 위치정보
        x_bin_def, y_bin_def = self._get_cell_indexes(
            players_def_coo[:, 0],
            players_def_coo[:, 1],
        )
        matrix[1, y_bin_def, x_bin_def] = 1
        
        # CH 3: Distance to ball
        # 모든 pixel에서 시작공까지 거리
        yy, xx = np.ogrid[0.5 : self.y_bins, 0.5 : self.x_bins]

        x0_ball, y0_ball = self._get_cell_indexes(ball_coo[:, 0], ball_coo[:, 1])
        matrix[2, :, :] = np.sqrt((xx - x0_ball) ** 2 + (yy - y0_ball) ** 2)

        # CH 4: Distance to goal
        # 모든 pixel에서 골대까지 거리
        x0_goal, y0_goal = self._get_cell_indexes(goal_coo[:, 0], goal_coo[:, 1])
        matrix[3, :, :] = np.sqrt((xx - x0_goal) ** 2 + (yy - y0_goal) ** 2)

        # CH 5: Cosine of the angle between the ball and goal
        # 모든 위치에서 시작공과 골대사이의 코사인값
        coords = np.dstack(np.meshgrid(xx, yy))
        goal_coo_bin = np.concatenate((x0_goal, y0_goal))
        ball_coo_bin = np.concatenate((x0_ball, y0_ball))
        a = goal_coo_bin - coords
        b = ball_coo_bin - coords
        matrix[4, :, :] = np.clip(
            np.sum(a * b, axis=2) / (np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2)), -1, 1
        )

        # CH 6: Sine of the angle between the ball and goal
        # sin = np.cross(a,b) / (np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2))
        # 모든 위치에서 시작공과 골대사이의 사인값
        matrix[5, :, :] = np.sqrt(1 - matrix[4, :, :] ** 2)  # This is much faster

        # CH 7: Angle (in radians) to the goal location
        # 모든 위치에서 골대까지 각도
        matrix[6, :, :] = np.abs(
            np.arctan((y0_goal - coords[:, :, 1]) / (x0_goal - coords[:, :, 0]))
        )

        # CH 8-9: Ball speed
        if speed_x<0 or speed_x>50:
            speed_x = 0
        if speed_y<0 or speed_y>50:
            speed_x = 0 
        # matrix[7, y0_ball, x0_ball] = speed_x
        # matrix[8, y0_ball, x0_ball] = speed_y   
        matrix[7, :, :] = speed_x
        matrix[8, :, :] = speed_y

        # # CH 9: Distance to ball_end
        # # 모든 pixel에서 도착지까지 거리
        # x0_ball_end, y0_ball_end = self._get_cell_indexes(ball_end_coo[:, 0], ball_end_coo[:, 1])
        # matrix[12, :, :] = np.sqrt((xx - x0_ball_end) ** 2 + (yy - y0_ball_end) ** 2)

        # # CH 10: Cosine of the angle between the ball_end and goal
        # # 모든 위치에서 도착지과 골대사이의 코사인값
        # coords = np.dstack(np.meshgrid(xx, yy))
        # goal_coo_bin = np.concatenate((x0_goal, y0_goal))
        # ball_end_coo_bin = np.concatenate((x0_ball_end, y0_ball_end))
        # a = goal_coo_bin - coords
        # b = ball_end_coo_bin - coords
        # matrix[13, :, :] = np.clip(
        #     np.sum(a * b, axis=2) / (np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2)), -1, 1
        # )

        # # CH 11: Sine of the angle between the ball and goal
        # # 모든 위치에서 도착지과 골대사이의 사인값
        # matrix[14, :, :] = np.sqrt(1 - matrix[10, :, :] ** 2)

        # matrix[9, :, :] = ball_height_ground
        # matrix[10, :, :] = ball_height_low
        # matrix[11, :, :] = ball_height_high
  
        for i in range(len(matrix)):
            matrix[i, :, :] = scaler.fit_transform(matrix[i, :, :])

        # Mask
        mask = np.zeros((1, self.y_bins, self.x_bins))


        #end_ball_coo = np.array([[intended_end_x, intended_end_y]])
        end_ball_coo = np.array([[end_x, end_y]])
        
        if np.isnan(end_ball_coo).any():
            raise ValueError("End coordinates not known.")
        x0_ball_end, y0_ball_end = self._get_cell_indexes(end_ball_coo[:, 0], end_ball_coo[:, 1])
        mask[0, y0_ball_end, x0_ball_end] = 1


        if "receiver" in sample:
            target = int(sample["receiver"]) if not math.isnan(sample["receiver"]) else -1
            if target==-1:
                print("pass selection에서 target이 -1이 존재합니다. 멈추겠습니다.")
                exit()         
            return (
                torch.from_numpy(matrix).float(),
                torch.from_numpy(mask).float(),
                torch.tensor([target]).float(),
            )
        
        return (
            torch.from_numpy(matrix).float(),
            torch.from_numpy(mask).float(),
            torch.tensor([1]).float(),
        )


#이미 만들어진 모델 + 새로운데이터 
#SoccerMap input형태에맞게 바꾼 후 모델에 적용하여
#원하는 pass_selection probability출력
class SoccerMapComponent(PassSelectionComponent, UnxPassPytorchComponent):
    """A SoccerMap deep-learning model."""

    def __init__(self, model: PytorchSoccerMapModel):
        super().__init__(
            model=model,
            features={
                "startlocation": ["start_x_a0", "start_y_a0"],
                "endlocation": ["end_x_a0", "end_y_a0"],
                "freeze_frame_360": ["freeze_frame_360_a0","freeze_frame_360_a1","freeze_frame_360_a2"],
                "speed": ["speedx_a02", "speedy_a02"],
                "intended_endlocation_function" : ['intended_end_x_a0','intended_end_y_a0'],
                "ball_height_onehot" : ["ball_height_ground_a0", "ball_height_low_a0", "ball_height_high_a0"]
            },
            #label = ["receiver"],# just a dummy lalel
            label=['success'],
            transform=ToSoccerMapTensor(dim=(68, 104)),
            #transform=soccermap_ToSoccerMapTensor(dim=(68, 104)),
        )

    def test(self, dataset, batch_size=32, num_workers=0, pin_memory=False, **test_cfg) -> Dict:
        # Load dataset
        data = self.initialize_dataset(dataset, train_test=False)

        dataloader = DataLoader(
            data,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # Switch to test mode
        torch.set_grad_enabled(False)
        self.model.eval()

        # Apply model on test set
        all_preds, all_targets = [], []
        for batch in track(dataloader,description="pass selection model testing"):
            x, mask, _ = batch
            surface = self.model(x)
            for i in range(x.shape[0]):
                teammate_locations = torch.nonzero(x[i, 0, :, :])
                if len(teammate_locations) > 0:
                    #SoccerMap이 예측한 패스 선택 확률 표면에서 각 팀원이 위치한 pixel에서 pass selection probability
                    p_teammate_selection = surface[
                        i, 0, teammate_locations[:, 0], teammate_locations[:, 1]
                    ]
                    #실제로 패스한 도착지와 가장 가까이 있는 선수를 target_receiver(정답)임
                    selected_teammate = torch.argmin(
                        torch.cdist(torch.nonzero(mask[i, 0]).float(), teammate_locations.float())
                    )
                    
                    #팀원중 패스선택확률이 제일 높은 팀원 인덱스 == 실제 정답인 팀원 인덱스가 같은지를 accracy로 측정
                    all_targets.append(
                        (torch.argmax(p_teammate_selection) == selected_teammate).item()
                    )
         
                else:
                    all_targets.append(True)

            #해당 masking된 예측값만을 사용
            y_hat = pixel(surface, mask)
            all_preds.append(y_hat)
        all_preds = torch.cat(all_preds, dim=0).detach().numpy()[:, 0]

        # Compute metrics
        # 정답값은 패스목적지가 1인 sparse matrix이므로 해당 목적지의 y값은 모두 1
        y = np.ones(len(all_preds))
        #cleaerprint(f"y.shape = {y.shape}, all_preds len = {len(all_preds)}")

        # return self._get_metrics(all_targets, all_preds)
        # dict1 = self._get_metrics(all_targets, all_preds)
        return  {
            "log_loss": log_loss(y, all_preds, labels=[0, 1]),
            # "brier2": brier_score_loss(y, all_preds),
            "Accuracy": sum(all_targets) / len(all_targets),
        }