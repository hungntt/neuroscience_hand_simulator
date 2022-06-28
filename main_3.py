from typing import Dict
from ray import tune

import numpy as np
import xgboost as xgb

from main_4 import TestDataset


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

    dtest = TestDataset(test_data_path="original_test_data/Test.npy").get_dataset()

    # Train the classifier
    results = {}
    xgb.train(
            param,
            dtrain,
            evals=[(dtest, "eval")],
            evals_result=results,
            verbose_eval=False)
    accuracy = 1. - results["eval"]["merror"][-1]
    tune.report(mean_accuracy=accuracy, done=True)


if __name__ == "__main__":
    # TODO: Parameters for XGBoost training
    param = {  # error evaluation for multiclass training
        'num_class': 6,
        'eta': tune.loguniform(1e-4, 4e-1),
        'eval_metric': 'merror',
        'max_depth': tune.randint(1, 15),
        'max_leaves': tune.randint(1, 15),
        'subsample': tune.uniform(0.5, 1.0),
        'num_boost_round': tune.randint(10, 100),
        'min_child_weight': tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    }

    analysis = tune.run(
            train_model,
            config=param,
            num_samples=10,
    )

    # # TODO: Save the model here (hint: it should be 1 line)
    # model_xgb.save_model("model.json")
