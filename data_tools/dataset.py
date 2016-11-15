import numpy as np


class Dataset:
    """
    Class storing training and test data.
    """

    def __init__(self, data, division_ratio=0.7):
        self.training_labels = []
        self.test_labels = []

        row_count = data.shape[0]
        training_data = data[:int(row_count * division_ratio), :]
        test_data = data[int(row_count * division_ratio):, :]

        for row in training_data:
            self.training_labels.append(row[0])
        for row in test_data:
            self.test_labels.append(row[0])

        self.training_data = np.delete(training_data, 0, 1)
        self.test_data = np.delete(test_data, 0, 1)
