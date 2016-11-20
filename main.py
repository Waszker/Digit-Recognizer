from data_tools.dataset_reader import DatasetReader
from data_tools.dataset import Dataset
from data_tools.normalization import Normalizer
from classifiers.classifier import Classifier
from classifiers.neural_network import ClassificationNetwork
import numpy as np


def _prepare_training_data():
    print 'Reading training data'
    reader = DatasetReader("Datasets")
    data = reader.read_training_csv_set()
    print 'Finished reading training data'

    # Getting proper class labels
    labels = []
    for image in data:
        labels.append(image.correct_class)
    labels = np.asarray(labels)
    labels = labels.reshape(-1, 1)

    # Get normalized data
    print "Normalizing data"
    normalizer = Normalizer(data)
    print 'Finished normalizing training data'
    normalized_data = normalizer.get_normalized_data_matrix(data)
    normalized_data = np.concatenate((labels, normalized_data), axis=1)
    print normalized_data.shape
    np.savetxt("Datasets/normalized_training_data.csv", normalized_data, delimiter=",")


def _run_classification():
    print "Reading data"
    reader = DatasetReader("Datasets")
    dataset = Dataset(reader.read_normalized_data_for_classifier(), division_ratio=0.7)

    # network = ClassificationNetwork(dataset.training_data, dataset.test_data, training_labels=dataset.training_labels)
    # print "Network error: " + str(network.train(iteration_count=10))

    print "Training classifier on: " + str(dataset.training_data.shape[0]) + " samples"

    classifier = Classifier(dataset)
    classifiers = ['svm', 'rf', 'knn', 'lr', 'br', 'llr']
    for c in classifiers:
        print "Starting classifier: " + str(c)
        classifier.train(c)

        print "Testing classifier " + str(c)
        predictions = classifier.test()

        index = 0
        positive_count = 0
        all_samples = 0
        for prediction in predictions:
            if dataset.test_labels[index] == prediction:
                positive_count += 1
            all_samples += 1
            index += 1
        print "All in all error rate for " + str(c) + " is: " + str(1.0 - float(positive_count) / all_samples)


if __name__ == "__main__":
    _prepare_training_data()
    _run_classification()
    # data[0].show(new_size=(300, 300))
    # data[1].show(new_size=(300, 300))
    # data[2].show(new_size=(300, 300))
