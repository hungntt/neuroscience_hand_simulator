import os
import sys
from typing import Dict

import numpy as np
import xgboost as xgb
from ray import tune
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler

from main_4 import TestDataset

sys.path.insert(0, "")


class TrainDataset:
    """Class that generates the training data.

    Attributes
    ----------
    class_datafile_map : Dict[int, str]
        A dictionary whose keys are the class labels and values are the paths to data.npy as strings.
    """

    def __init__(self, class_datafile_map: Dict[int, str]):
        self.class_datafile_map = class_datafile_map

    def get_dataset(self) -> xgb.DMatrix:
        """Returns the dataset as a xgboost.DMatrix.

        Returns
        -------
        xgb.DMatrix
            The returned dataset.
        """
        train_data, train_labels = [], []
        for class_label, file_path in self.class_datafile_map.items():
            emg = np.load(file_path)
            train_data.append(emg.T)

            train_labels += [class_label] * emg.shape[1]

        train_data = np.concatenate(np.array(train_data), axis=0)
        train_labels = np.array(train_labels)
        print(train_data.shape, train_labels.shape)
        return xgb.DMatrix(data=train_data, label=train_labels)


def train_model(param):
    dtrain = TrainDataset(
            class_datafile_map={
                # TODO: load your training data from the train_data/ folder
                0: "original_train_data/Thumb.npy",
                1: "original_train_data/Index.npy",
                2: "original_train_data/Middle.npy",
                3: "original_train_data/Ring.npy",
                4: "original_train_data/Pinky.npy",
                5: "original_train_data/Fist.npy",
            }
    ).get_dataset()

    dtest = TestDataset(
            test_data_path="original_test_data/Test.npy").get_dataset()

    # Train the classifier
    results = {}
    xgb.train(
            param,
            dtrain,
            evals=[(dtest, "eval")],
            evals_result=results,
            verbose_eval=False,
            callbacks=[TuneReportCheckpointCallback(filename="model.xgb")])


def get_best_model_checkpoint(analysis):
    best_bst = xgb.Booster()
    best_bst.load_model(os.path.join(analysis.best_checkpoint, "model.xgb"))
    accuracy = 1. - analysis.best_result["eval-merror"]
    print(f"Best model parameters: {analysis.best_config}")
    print(f"Best model total accuracy: {accuracy:.4f}")
    return best_bst


def tune_xgboost():
    param = {  # error evaluation for multiclass training
        'objective': 'multi:softmax',
        'num_class': 6,
        'eta': tune.loguniform(2e-1, 6e-1),
        'eval_metric': ['mlogloss', 'merror'],
        'max_depth': tune.randint(25, 30),
        'max_leaves': tune.randint(1, 15),
        'subsample': tune.uniform(0.5, 1.0),
        'min_child_weight': tune.choice([1, 2, 3, 4, 5]),
        'verbosity': tune.randint(0, 4),
    }

    scheduler = ASHAScheduler(
            max_t=10,  # 10 training iterations
            grace_period=1,
            reduction_factor=2)

    analysis = tune.run(
            train_model,
            metric="eval-mlogloss",
            mode="min",
            resources_per_trial={"cpu": 1, "gpu": 0.125},
            config=param,
            num_samples=30,
            scheduler=scheduler,
    )
    return analysis


if __name__ == "__main__":
    analysis = tune_xgboost()
    best_bst = get_best_model_checkpoint(analysis)
    print(best_bst)
