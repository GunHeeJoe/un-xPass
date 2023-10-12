"""Model architectures."""
import pickle
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union
import tqdm
import mlflow
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import xgboost as xgb
from rich.progress import track
import time
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
import random
from torch.utils.data import DataLoader, Subset, random_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import log_loss
from unxpass.config import logger as log
from unxpass.datasets import PassesDataset
from unxpass.features import simulate_features
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from unxpass.sampler import SubsetRandomSampler_function, WeightedRandomSampler_function,My_Sampler

class UnxpassComponent(ABC):
    """Base class for all components."""

    component_name = "default"

    def __init__(
        self, features: Union[List, Dict], label: List, transform: Optional[Callable] = None
    ):
        self.features = features
        self.label = label
        self.transform = transform

    def initialize_dataset(self,dataset: Union[PassesDataset, Callable], train_test=True) -> PassesDataset:
        if callable(dataset):
            return dataset(xfns=self.features, yfns=self.label, transform=self.transform,train_test=train_test)
        return dataset

    @abstractmethod
    def train(self, dataset: Callable, optimized_metric=None) -> Optional[float]:
        pass

    @abstractmethod
    def test(self, dataset: Callable) -> Dict[str, float]:
        pass

    def _get_metrics(self, y_true, y_hat):
        return {}

    @abstractmethod
    def predict(self, dataset: Callable) -> pd.Series:
        pass

    def save(self, path: Path):
        pickle.dump(self, path.open(mode="wb"))

    @classmethod
    def load(cls, path: Path):
        return pickle.load(path.open(mode="rb"))
    
    #samples : predict 
    #true_labels : target
    def expected_calibration_error(self, samples, true_labels, M=10):
        # uniform binning approach with M number of bins
        #(0, 0.1), (0.1, 0.2), (0.2, 0.3)......(0.9, 1.0)의 (lower, upper)쌍이 나옴
        bin_boundaries = np.linspace(0, 1, M + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        # keep confidences / predicted "probabilities" as they are
        confidences = samples
        # get binary class predictions from confidences
        predicted_label = (samples>0.5).astype(float)

        # get a boolean list of correct/false predictions
        accuracies = predicted_label==true_labels
        positive_class = true_labels==1

        soccermap_ece = np.zeros(1)
        epv_ece = np.zeros(1)
        soccermap_ece1 = np.zeros(1)
        epv_ece1 = np.zeros(1)

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # determine if sample is in bin m (between bin lower & upper)
            # confidence가 지정된 값 사이의 존재하는지 여뷰 
            in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
            
            # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
            # 전체 데이터 중 해당 Bin에 해당하는 확류ㅠㄹ
            prop_in_bin = in_bin.astype(float).mean()
            
            if prop_in_bin.item() > 0:
                # get the accuracy of bin m: acc(Bm)
                #accuracy
                accuracy_in_bin = accuracies[in_bin].astype(float).mean()
                #predict
                label_in_bin = predicted_label[in_bin].astype(float).mean()

                #둘다 confidence=예측값을 활용하고, 추가로
                #epv'ece는 true_label의 평균을 활용하고
                ground_truth = true_labels[in_bin].astype(float).mean()
                #soccermap'ece는 (true_label=1)의 평균을 활용함
                positive_class_in_bin = positive_class[in_bin].astype(float).mean()

                # get the average confidence of bin m: conf(Bm)
                avg_confidence_in_bin = confidences[in_bin].mean()

                # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
                # soccermap_ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                # epv_ece += np.abs(avg_confidence_in_bin - label_in_bin) * prop_in_bin

                soccermap_ece1 += np.abs(avg_confidence_in_bin - positive_class_in_bin) * prop_in_bin
                epv_ece1 += np.abs(avg_confidence_in_bin - ground_truth) * prop_in_bin
                #print(f"{(bin_lower, bin_upper)} -> {(avg_confidence_in_bin,accuracy_in_bin)} => {(avg_confidence_in_bin-accuracy_in_bin)}")
            
        # print(f"original used evalution : soccermap'ece = {soccermap_ece}, epv'ece = {epv_ece}")
        # print(f"new used evalution : soccermap'ece1 = {soccermap_ece1}, epv'ece1 = {epv_ece1}")
        return soccermap_ece1, epv_ece1


class UnxPassSkLearnComponent(UnxpassComponent):
    """Base class for an SkLearn-based component."""

    def __init__(self, model, features, label):
        super().__init__(features, label)
        
        self.model = model
    def train(self, dataset, optimized_metric='neg_log_loss') -> Optional[float]:
        #mlflow.sklearn.autolog()
             
        # Load data
        data = self.initialize_dataset(dataset)
        X_train, y_train = data.features, data.labels

        # for col in X_train.columns:
        #     print(col, " -> ",X_train[col].dtypes)
        # ss

        X_train = StandardScaler().fit_transform(X_train)
        
        # parameters = {'learning_rate_init':[1e-3,1e-4,1e-5,1e-6], 
        #               'batch_size':[1, 16, 32, 64]}
             
        # self.model = GridSearchCV(self.model, param_grid=parameters, 
        #                           cv=3,scoring=optimized_metric,verbose=0)
        # Train the model
        log.info("Fitting model on train set")
        self.model.fit(X_train, y_train)

        #print("best parameter : ",self.model.best_params_)
        return None

    def test(self, dataset) -> Dict[str, float]:
        data = self.initialize_dataset(dataset)
        X_test, y_test = data.features, data.labels
        
        #X_test = pd.get_dummies(X_test)
        X_test = StandardScaler().fit_transform(X_test)
        
        y_hat = self.model.predict_proba(X_test)[:, 1]
        return self._get_metrics(y_test, y_hat)

    def predict(self, dataset) -> pd.Series:
        data = self.initialize_dataset(dataset)
        y_hat = self.model.predict_proba(data.features)[:, 1]
        return pd.Series(y_hat, index=data.features.index)


class UnxPassXGBoostComponent(UnxpassComponent):
    """Base class for an XGBoost-based component."""

    def __init__(self, model, features, label):
        super().__init__(features, label)
        self.model = model
        self.prohibit_features = []
    def train(self, dataset, optimized_metric='logloss', **train_cfg) -> Optional[float]:
        #mlflow.xgboost.autolog()
        self.model.set_params(verbosity=0)
        # Load data
        data = self.initialize_dataset(dataset)

        X_train, X_val, y_train, y_val = train_test_split(
            data.features, data.labels, stratify=data.labels,test_size=0.2
        )

        print_col = ['ball_height_ground_a0','ball_height_low_a0','ball_height_high_a0']

        for col in X_train.columns:
            if (X_train[col].dtypes != 'object') and ("result" not in col):
                self.prohibit_features.append(col)
                
        X_train = X_train[self.prohibit_features]
        X_val = X_val[self.prohibit_features]
        
        X_train = pd.DataFrame(StandardScaler().fit_transform(X_train),columns=X_train.columns)              
        X_val = pd.DataFrame(StandardScaler().fit_transform(X_val),columns=X_val.columns)
        
        # parameters = {
        #     'num_round' : [100,500,1000],
        #     'max_depth': [1,9],
        #     'eta': [1e-2, 0.25],
        #     'alpha': [1e-8, 100],
        #     'lambda': [1e-8, 100],
        #     'gamma': [1e-8, 1],
        #     'early_stopping_rounds': [100],
        #     'eval_metric': ['logloss'],
        #     'objective': ['binary:logistic'],
        # }
                
        # self.model = GridSearchCV(self.model, param_grid=parameters, 
        #                 cv=3,scoring='neg_log_loss',verbose=0)
        # Train the model
        log.info("Fitting model on train set")
        
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], **train_cfg)

        # Return metric score for hyperparameter optimization
        # if optimized_metric is not None:
        #     idx = self.model.best_iteration
        #     return self.model.evals_result()["validation_0"][optimized_metric][idx]
        #print("best parameter : ",self.model.best_params_)
        return None

    def test(self, dataset) -> Dict[str, float]:
        data = self.initialize_dataset(dataset)
        X_test, y_test = data.features, data.labels
                
        X_test = X_test[self.prohibit_features]
        X_test = pd.DataFrame(StandardScaler().fit_transform(X_test),columns=X_test.columns)
        
        y_hat = self.model.predict_proba(X_test)[:,1]
        if isinstance(self.model, xgb.XGBClassifier):
            y_hat = self.model.predict_proba(X_test)[:, 1]
        elif isinstance(self.model, xgb.XGBRegressor):
            y_hat = self.model.predict(X_test)
        else:
            raise AttributeError(f"Unsupported xgboost model: {type(self.model)}")
        return self._get_metrics(y_test, y_hat)

    def predict(self, dataset) -> pd.Series:
        data = self.initialize_dataset(dataset)
        X_test, y_test = data.features, data.labels
        y_hat = self.model.predict_proba(X_test)[:,1]
        # if isinstance(self.model, xgb.XGBClassifier):
        #     y_hat = self.model.predict_proba(data.features)[:, 1]
        # elif isinstance(self.model, xgb.XGBRegressor):
        #     y_hat = self.model.predict(data.features)
        # else:
        #     raise AttributeError(f"Unsupported xgboost model: {type(self.model)}")
        return pd.Series(y_hat, index=data.features.index)

    def predict_locations(self, dataset, game_id, db, xy_coo, result=None) -> pd.Series:
        data = self.initialize_dataset(dataset)
        games = data.features.index.unique(level=0)
        assert game_id in games, "Game ID not found in dataset!"
        sim_features = simulate_features(
            db,
            game_id,
            xfns=list(data.xfns.keys()),
            actionfilter=data.actionfilter,
            xy=xy_coo,
            result=result,
        )
        X_test, y_test = data.features, data.labels
        cols = [item for sublist in data.xfns.values() for item in sublist]
        y_hat = self.model.predict_proba(X_test)[:,1]
        # if isinstance(self.model, xgb.XGBClassifier):
        #     y_hat = self.model.predict_proba(sim_features[cols])[:, 1]
        # elif isinstance(self.model, xgb.XGBRegressor):
        #     y_hat = self.model.predict(sim_features[cols])
        # else:
        #     raise AttributeError(f"Unsupported xgboost model: {type(self.model)}")
        return pd.Series(y_hat, index=data.features.index)

    def predict_surface(
        self, dataset, game_id, config=None, db=None, x_bins=104, y_bins=68, result=None
    ) -> Dict:
        data = self.initialize_dataset(dataset)
        games = data.features.index.unique(level=0)
        assert game_id in games, "Game ID not found in dataset!"
        sim_features = simulate_features(
            db,
            game_id,
            xfns=list(data.xfns.keys()),
            actionfilter=data.actionfilter,
            x_bins=x_bins,
            y_bins=y_bins,
            result=result,
        )
        print(sim_features)
        ss
        out = {}
        cols = [item for sublist in data.xfns.values() for item in sublist]
        for action_id in sim_features.index.unique(level=1):
            if isinstance(self.model, xgb.XGBClassifier):
                out[f"action_{action_id}"] = (
                    self.model.predict_proba(sim_features.loc[(game_id, action_id), cols])[:, 1]
                    .reshape(x_bins, y_bins)
                    .T
                )
            elif isinstance(self.model, xgb.XGBRegressor):
                out[f"action_{action_id}"] = (
                    self.model.predict(sim_features.loc[(game_id, action_id), cols])
                    .reshape(x_bins, y_bins)
                    .T
                )
            else:
                raise AttributeError(f"Unsupported xgboost model: {type(self.model)}")
        return out
    
#모델의 학습을 담당하는 클래스
#epoch나 batch등의 상태를 저장해서 모델링하는 코드
class UnxPassPytorchComponent(UnxpassComponent):
    """Base class for a PyTorch-based component."""

    def __init__(self, model, features, label, transform):
        super().__init__(features, label, transform)
        self.model = model.to('cuda:0')
        logger = TensorBoardLogger("../runs/", name="End11_pass_selection")
        early_stop_callback = EarlyStopping(monitor='loss/val', min_delta=1e-3,
                                            mode='min',patience=4)
        self.trainer = pl.Trainer(max_epochs=30,callbacks=early_stop_callback,
                             logger=logger,
                             accelerator="cuda",devices=1)#2, strategy='ddp')
        
    def train(
        self,
        dataset,
        optimized_metric=None,
        callbacks=None,
        logger=None,
        batch_size=1,
        num_workers=8,
        pin_memory=True,
        **train_cfg,
    ) -> Optional[float]:
        #mlflow.pytorch.autolog()
        #mlflow.pytorch.autolog(disable=True)
        # Init lightning trainer

        #trainer = pl.Trainer(callbacks=callbacks, logger=logger, **train_cfg["trainer"])
        
        #EarlyStopping parameter
        #monitor='val_loss',  # 모니터링할 지표: 검증 손실
        #min_delta=0,         # 최소 변화량 (기본값 0)
        #patience=3,          # 성능 개선이 없을 때 허용할 에포크 수 (기본값 3)
        #mode='min',          # 손실 값이 감소하는 방향으로 개선 여부를 판단
        #strict=True,         # 지정된 지표보다 성능이 개선되어야 조기 종료 허용
        #check_finite=True,   # 손실 값이 유효한지 확인
        #stop_on_nan=False,   # 손실 값이 NaN인 경우 훈련 종료하지 않음
        #verbose=False,       # 조기 종료 시 메시지를 표시하지 않음
        #check_interval='epoch'  # 에포크마다 조기 종료 확인
        # early_stop_callback = EarlyStopping(monitor='loss/val', min_delta=1e-3,
        #                                     mode='min',patience=3)
        # trainer = pl.Trainer(max_epochs=1,callbacks=early_stop_callback,
        #                      logger=self.logger,
        #                      accelerator="cuda", devices=2, strategy="ddp")
    
        # Load data
        data = self.initialize_dataset(dataset,train_test=True)
        
        # nb_train = int(len(data) * 0.8)
        # lengths = [nb_train, len(data) - nb_train]
        # _data_train, _data_val = random_split(data, lengths)

        # train_dataloader = DataLoader(
        #     _data_train,
        #     batch_size=batch_size,
        #     num_workers=num_workers,
        #     pin_memory=pin_memory,
        #     shuffle=True,
        # )
        # val_dataloader = DataLoader(
        #     _data_val,
        #     batch_size=batch_size,
        #     num_workers=num_workers,
        #     pin_memory=pin_memory,
        #     shuffle=False,
        # )
        #train : valid = success : fail비율을 같게 sampling

        # train : valid = success : fail비율을 같게 sampling
        # train batch마다 success : fail 비율을 같게 sampling
        #train_data, valid_data, train_sampler = My_Sampler(data)
        train_sampler, valid_sampler = SubsetRandomSampler_function(data)

        train_dataloader = DataLoader(
            data,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler = train_sampler
        )

        val_dataloader = DataLoader(
            data,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sampler = valid_sampler
        )

        # Train the model
        #log.info("Fitting model on train set")
        
        
        self.trainer.fit(
            model=self.model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )
        
        # Print path to best checkpoint
        #log.info(f"Best checkpoint path:\n{trainer.checkpoint_callback.best_model_path}")

        # Return metric score for hyperparameter optimization
        if optimized_metric is not None:
            return self.trainer.callback_metrics[optimized_metric]

        return None

    def test(
        self, dataset, batch_size=32, num_workers=8, pin_memory=True, **test_cfg
    ) -> Dict[str, float]:
        # Load dataset
        data = self.initialize_dataset(dataset)
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
        # for batch in track(dataloader,description="pass success model testing"):
        for batch in tqdm.tqdm(dataloader,desc="pass success model testing"):
            loss, y_hat, y = self.model.step(batch)
            all_preds.append(y_hat)
            all_targets.append(y)
        all_preds = torch.cat(all_preds, dim=0).detach().numpy()[:, 0]
        all_targets = torch.cat(all_targets, dim=0).detach().numpy()[:, 0]

        # Compute metrics
        return self._get_metrics(all_targets, all_preds)
        
    def predict(self, dataset, batch_size=32, num_workers=0, pin_memory=False) -> pd.Series:
        # Load dataset
        data = self.initialize_dataset(dataset)
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
        all_preds = []
        for batch in track(dataloader):
            loss, y_hat, y = self.model.step(batch)
            all_preds.append(y_hat)
        all_preds = torch.cat(all_preds, dim=0).detach().numpy()[:, 0]
        return pd.Series(all_preds, index=data.features.index)

    def predict_surface(
        self, dataset, game_id=None, batch_size=32, num_workers=0, pin_memory=False, **predict_cfg
    ) -> Dict:
        # Load dataset
        data = self.initialize_dataset(dataset)
        actions = data.features.reset_index()
        if game_id is not None:
            actions = actions[actions.game_id == game_id]
            data = Subset(data, actions.index.values)
        dataloader = DataLoader(
            data,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        #predict_surface는 인덱스에 맞는 예측값을 output에 넣는거라서 
        #gpu 2개 써버리면 index가 안맞는 형식으로 출력이 되버림
        predictor = pl.Trainer(**predict_cfg.get("trainer", {}),devices=1)
        predictions = torch.cat(predictor.predict(self.model, dataloaders=dataloader))

        output = defaultdict(dict)
        for i, action in actions.iterrows():
            output[action.game_id][action.action_id] = predictions[i][0].detach().numpy()

        return dict(output)

    @classmethod
    def load(cls, path: Path):
        return pickle.load(path.open(mode="rb"))
