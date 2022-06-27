from typing import Dict

import numpy as np
import xgboost as xgb


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


if __name__ == "__main__":
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

    # TODO: Parameters for XGBoost training
    param = {  # error evaluation for multiclass training
        'num_class': 6,
        'eta': 0.33,
        'objective': 'multi:softmax',
    }

    model_xgb = xgb.train(params=param, dtrain=dtrain, num_boost_round=100)

    # TODO: Save the model here (hint: it should be 1 line)
    model_xgb.save_model(f"model_1.json")
