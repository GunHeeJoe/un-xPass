"""Implements the pass success probability component."""
from typing import Any, Dict, List
from torch.utils.tensorboard import SummaryWriter
import mlflow
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from mplsoccer import Pitch

from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from .base import (
    UnxpassComponent,
    UnxPassPytorchComponent,
    UnxPassSkLearnComponent,
    UnxPassXGBoostComponent,
)
from .soccermap import SoccerMap, pixel, soccermap_ToSoccerMapTensor
from torchmetrics.classification import BinaryCalibrationError


class PassSuccessComponent(UnxpassComponent):
    """The pass success probability component.

    From any given game situation where a player controls the ball, the model
    estimates the success probability of a pass attempted towards a potential
    destination location.
    """

    component_name = "pass_success"

    def _get_metrics(self, y, y_hat):
        #BenchMark_Model(LogisticNet, DenseNet)에서 사용하는 코드
        #EPV_Model에서는 사용하지않음
        #y = y['success']
        bce_l1 = BinaryCalibrationError(n_bins=10,norm='l1')
        tensor_y_hat =  torch.Tensor(y_hat)
        tensor_y =  torch.Tensor(y)

        soccermap_ece, epv_ece = self.expected_calibration_error(y_hat,y)
        y_pred = y_hat > 0.5
        return {
            "precision": precision_score(y, y_pred),
            "recall": recall_score(y, y_pred),
            "f1": f1_score(y, y_pred),
            "log_loss": log_loss(y, y_hat),
            "brier": brier_score_loss(y, y_hat),
            "roc_auc": roc_auc_score(y, y_hat),
            "soccermap_ece" : soccermap_ece,
            "epv_ece" : epv_ece,
            "bce_l1" : bce_l1(tensor_y_hat, tensor_y),
        }


class NaiveBaselineComponent(UnxPassSkLearnComponent, PassSuccessComponent):
    """A baseline model that assigns the average pass completion to all passes."""

    def __init__(self):
        super().__init__(
            model=DummyClassifier(strategy="prior"),
            features={"startlocation": ["start_x_a0"]},  # a dummy feature
            label=["success"],
        )

class SklearnComponent(PassSuccessComponent, UnxPassSkLearnComponent):
    """A XGBoost model based on handcrafted features."""

    def __init__(
        self, model, features: Dict[str, List[str]], label: List[str] = ["success"]
    ):
        super().__init__(
            model=model,
            features=features,
            label=label,
        )
        
class XGBoostComponent(PassSuccessComponent, UnxPassXGBoostComponent):
    """A XGBoost model based on handcrafted features."""

    def __init__(
        self, model: XGBClassifier, features: Dict[str, List[str]], label: List[str] = ["success"]
    ):
        super().__init__(
            model=model,
            features=features,
            label=label,
        )

#nn.Module처럼 모델 내부의 구조를 설계
class PytorchSoccerMapModel(pl.LightningModule):
    """A pass success probability model based on the SoccerMap architecture."""

    def __init__(
        self,
        #lr: float = 1e-4,
        lr : float = 1e-5
    ):
        super().__init__()
        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        #self.save_hyperparameters()

        #self.model = SoccerMap(in_channels=7)
        self.model = SoccerMap(in_channels=9)
        self.sigmoid = nn.Sigmoid()

        # loss function
        self.criterion = torch.nn.BCELoss()

    def forward(self, x: torch.Tensor):
        x = self.model(x)

        x = self.sigmoid(x)

        return x

    def step(self, batch: Any):
        x, mask, y = batch

        surface = self.forward(x)
    
        #해당 목적지의 단일 예측값만 loss로 사용

        y_hat = pixel(surface, mask)
        
        loss = self.criterion(y_hat, y)

        return loss, y_hat, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        #print(torch.unique(targets,return_counts=True))
        
        # log train metrics
        # name : 로그이름
        # value : 기록할 값
        # on_step : 훈련스탭마다 로그 기록
        # prog_bar : 진행 막대 표시
        # logger : 로그 기록을 위한 looger
        # sync_dist : 분산 환경에서 로그 동기화
        self.log("loss/train", loss, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        #self.logger.experiment.add_scalar("train/loss",loss)
        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        self.log("loss/val", loss, on_step=False, on_epoch=True, prog_bar=True,sync_dist=True)
        #self.logger.experiment.add_scalar("val/loss",loss)
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        #self.log("loss/test", loss, on_step=False, on_epoch=True,prog_bar=True,sync_dist='auto')
        
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
        # return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)
        return torch.optim.Adam(params=self.parameters(), lr=1e-4)
    
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

        scaler = StandardScaler()
        speed_x, speed_y = sample["speedx_a02"], sample["speedy_a02"]
        frame = pd.DataFrame.from_records(sample["freeze_frame_360_a0"])
        #success model은 도착한 패스의 pixel부분의 성공여부로 학습함
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

        #겹치는 선수가 존재할때 1로 간주할 것인가 그 이상으로 간주할것인가
        # for i in range(len(x_bin_att)):
        #     matrix[0, y_bin_att[i], x_bin_att[i]] += 1
        matrix[0, y_bin_att, x_bin_att] = 1

        # CH 2: Locations of defending team
        # 수비팀의 위치정보
        x_bin_def, y_bin_def = self._get_cell_indexes(
            players_def_coo[:, 0],
            players_def_coo[:, 1],
        )
        # for i in range(len(x_bin_def)):
        #     matrix[0, y_bin_def[i], x_bin_def[i]] += 1
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
        # if speed_x<0 or speed_x>50:
        #     speed_x = 0
        # if speed_y<0 or speed_y>50:
        #     speed_x = 0 
        # matrix[7, y0_ball, x0_ball] = speed_x
        # matrix[8, y0_ball, x0_ball] = speed_y   
        matrix[7, :, :] = speed_x
        matrix[8, :, :] = speed_y

        # CH 9: Distance to ball_end
        # 모든 pixel에서 도착지까지 거리
        #도착지 ball_end_coo정보를 사용하는건 옮지 않음
        #우리는 단순히 XGBOost처럼 start->end로 가는 패스 성공확률을 추정하는 것이 아님
        #모든 pixel로 향하는 패스 성공 확률을 추정하고자 CNN를 접목시킨건데, channel=end정보를 넣어버리면
        #특정 상황에서만 패스 성공확률을 잘 맞추는 문제가 생겨버리므로 사용해서는 안되는 정보임
        # x0_ball_end, y0_ball_end = self._get_cell_indexes(ball_end_coo[:, 0], ball_end_coo[:, 1])
        # matrix[9, :, :] = np.sqrt((xx - x0_ball_end) ** 2 + (yy - y0_ball_end) ** 2)

        # # CH 10: Cosine of the angle between the ball_end and goal
        # # 모든 위치에서 도착지과 골대사이의 코사인값
        # coords = np.dstack(np.meshgrid(xx, yy))
        # goal_coo_bin = np.concatenate((x0_goal, y0_goal))
        # ball_end_coo_bin = np.concatenate((x0_ball_end, y0_ball_end))
        # a = goal_coo_bin - coords
        # b = ball_end_coo_bin - coords
        # matrix[10, :, :] = np.clip(
        #     np.sum(a * b, axis=2) / (np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2)), -1, 1
        # )

        # # CH 11: Sine of the angle between the ball and goal
        # # 모든 위치에서 도착지과 골대사이의 사인값
        # matrix[11, :, :] = np.sqrt(1 - matrix[10, :, :] ** 2)

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


        print(f"end : ({y0_ball_end, x0_ball_end})")
        print(f"Intended : ({Intended_y0_ball_end, Intended_x0_ball_end})")
        if target is not None:
            return (
                torch.from_numpy(matrix).float(),
                torch.from_numpy(mask).float(),
                torch.tensor([target]).float(),
            )
        return (
            torch.from_numpy(matrix).float(),
            torch.from_numpy(mask).float(),
            None,
        )



class SoccerMapComponent(PassSuccessComponent, UnxPassPytorchComponent):
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
            label=["success"],
            transform=ToSoccerMapTensor(dim=(68, 104)),
            #transform=soccermap_ToSoccerMapTensor(dim=(68, 104)),
        )
