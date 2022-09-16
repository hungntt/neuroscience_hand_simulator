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


# def train_model(param):
#
#
#     # Train the classifier
#     results = {}
#     xgb.train(
#             param,
#             dtrain,
#             evals=[(dtest, "eval")],
#             evals_result=results,
#             verbose_eval=False)
#     accuracy = 1. - results["eval"]["error"][-1]
#     tune.report(mean_accuracy=accuracy, done=True)


if __name__ == "__main__":
    # TODO: Parameters for XGBoost training
    param = {  # error evaluation for multiclass training
        'num_class': 6,
        'objective': 'multi:softmax',
        # 'eta': tune.loguniform(1e-4, 1e-1),
        'eval_metric': 'merror',
        # 'max_depth': tune.randint(1, 10),
        # 'subsample': tune.uniform(0.5, 1.0),
        # 'min_child_weight': tune.choice([1, 2, 3])
    }

    dtrain = TrainDataset(
            class_datafile_map={
                # TODO: load your training data from the train_data/ folder
                0: "train_data/Thumb.npy",
                1: "train_data/Index.npy",
                2: "train_data/Middle.npy",
                3: "train_data/Ring.npy",
                4: "train_data/Pinky.npy",
                5: "train_data/Fist.npy",
            }
    ).get_dataset()

    dtest = TestDataset(test_data_path="test_data/Test.npy").get_dataset()

    results = {}

    xgb.train(
            param,
            dtrain,
            evals=[(dtest, "eval")],
            evals_result=results,
            verbose_eval=False)
    accuracy = results["eval"]["merror"][-1]
    print(accuracy)
    # # TODO: Save the model here (hint: it should be 1 line)
    # model_xgb.save_model("model.json")
