from typing import Dict

import numpy as np
import xgboost as xgb
from tqdm import tqdm


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
        return xgb.DMatrix(data=train_data, label=train_labels)


def xgb_progressbar(rounds=1000):
    """Progressbar for xgboost using tqdm library."""

    pbar = tqdm(total=rounds)

    def callback(_, ):
        pbar.update(1)

    return callback


if __name__ == "__main__":
    dtrain = TrainDataset(
            class_datafile_map={
                # TODO: load your training data from the train_data/ folder
                0: "/home/ubuntu/neuroscience_hand_simulator/original_train_data/Thumb.npy",
                1: "/home/ubuntu/neuroscience_hand_simulator/original_train_data/Index.npy",
                2: "/home/ubuntu/neuroscience_hand_simulator/original_train_data/Middle.npy",
                3: "/home/ubuntu/neuroscience_hand_simulator/original_train_data/Ring.npy",
                4: "/home/ubuntu/neuroscience_hand_simulator/original_train_data/Pinky.npy",
                5: "/home/ubuntu/neuroscience_hand_simulator/original_train_data/Fist.npy",
            }
    ).get_dataset()

    # TODO: Parameters for XGBoost training
    param = {  # error evaluation for multiclass training
        'num_class': 6,
        'objective': 'multi:softmax',
        'max_depth': 29,
        'verbosity': 1,
        'eta': 0.3328483951122373,
        'min_child_weight': 5,
        'subsample': 0.626656143893046,
    }

    model_xgb = xgb.train(params=param, dtrain=dtrain, num_boost_round=50, verbose_eval=True)

    # TODO: Save the model here (hint: it should be 1 line)
    model_xgb.save_model(f"model_4.json")
