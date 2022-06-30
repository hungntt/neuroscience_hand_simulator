import numpy as np
import xgboost as xgb
from sklearn.metrics import precision_score


class TestDataset:
    """Class that generates the training data.

    Attributes
    ----------
    test_data_path : str
        A string that points to the test folder
    """

    def __init__(self, test_data_path: str):
        self.test_data_path = test_data_path

    def get_dataset(self) -> xgb.DMatrix:
        """Returns the dataset as a xgboost.DMatrix.

        Returns
        -------
        xgb.DMatrix
            The returned dataset.
        """

        # TODO: Load the test data and the labels with np.load and set them correctly.
        loaded_test_dataset = np.load(self.test_data_path)
        test_data = loaded_test_dataset[0:320, :].T
        test_labels = loaded_test_dataset[320, :].T
        print(test_data.shape, test_labels.shape)
        return xgb.DMatrix(data=test_data, label=test_labels)


if __name__ == "__main__":
    # TODO: Create a TestDataset object and call get_dataset() to save the test data.
    test_dataset = TestDataset(test_data_path="original_test_data/Test.npy").get_dataset()

    # TODO: Load your saved xgboost model
    model = xgb.Booster()
    model.load_model("model_2.json")

    # TODO: Predict the output of the model on the testing data and save it.
    predictions = model.predict(test_dataset)
    # Score the predictions
    print(precision_score(test_dataset.get_label(), predictions, average='macro'))

    # This code calculates the overall accuracy. Please aim for > 75%.
    count = 0
    for p, gt in zip(predictions, test_dataset.get_label()):
        if p == gt:
            count += 1

    print(count / len(predictions))
