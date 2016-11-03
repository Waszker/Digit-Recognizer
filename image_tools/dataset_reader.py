import os
from numpy import genfromtxt
from image import Image


class DatasetReader:
    """
    Takes care of reading data sets for training and testing.
    """

    def __init__(self, dataset_path):
        self.path = dataset_path

    def read_training_csv_set(self, filename='train.csv'):
        """
        Reads data from training file in .csv format.
        :param filename: optional parameter if filename is different than 'train.csv'
        :return: list containing read images
        """
        data = genfromtxt(self.path + os.sep + filename, delimiter=',', skip_header=1)
        images = []
        for row in data:
            images.append(Image(row[1:], correct_class=row[0]))

        return images

    def read_test_csv_set(self, filename='test.csv'):
        """
        Reads data from test file in .csv format.
        :param filename: optional parameter if filename is different than 'test.csv'
        :return: list containing read images
        """
        data = genfromtxt(self.path + os.sep + filename, delimiter=',', skip_header=1)
        images = []
        for row in data:
            images.append(Image(row))

        return images
