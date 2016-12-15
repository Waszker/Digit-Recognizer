from data_tools.dataset_reader import DatasetReader
from data_tools.dataset import Dataset
from data_tools.normalization import Normalizer
from classifiers.classifier import Classifier
from classifiers.neural_network import SoftmaxNetwork
from classifiers.backpropagation import Backpropagation
import numpy as np
import time


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
    start_time = time.time()
    normalizer = Normalizer(data)
    print 'Finished getting normalization vectors ' + str(time.time() - start_time)
    start_time = time.time()
    normalized_data = normalizer.get_normalized_data_matrix(data)
    print 'Finished normalizing training data in ' + str(time.time() - start_time)
    normalized_data = np.concatenate((labels, normalized_data), axis=1)
    print normalized_data.shape
    np.savetxt("Datasets/normalized_training_data.csv", normalized_data, delimiter=",")


def _run_classification():
    print "Reading data"
    reader = DatasetReader("Datasets")
    dataset = Dataset(reader.read_normalized_data_for_classifier(), division_ratio=0.7)
    print "Training classifier on: " + str(dataset.training_data.shape) + " samples"
    distribution1, distribution2 = dataset.get_classes_distribution()
    print "Class distribution for training data: " + str(distribution1)
    print "Class distribution for test data: " + str(distribution2)

    # TensorFlow tests
    network = SoftmaxNetwork(dataset.training_data, dataset.test_data, training_labels=dataset.training_labels,
                             test_labels=dataset.test_labels)
    network.train(iteration_count=2000)
    [accuracy, predictions] = network.test()
    index, all_samples_v, correct_samples_v = 0, np.array([0.] * 10), np.array([0.] * 10)
    for prediction in predictions:
        all_samples_v[prediction] += 1
        if dataset.test_labels[index] == prediction:
            correct_samples_v[prediction] += 1
        index += 1
    print "All samples: " + str(all_samples_v)
    print "Samples result vector: " + str(correct_samples_v / all_samples_v)
    print "Error rate for SoftmaxNetwork is: " + str(1 - accuracy)

    network = Backpropagation(dataset.training_data, dataset.test_data, training_labels=dataset.training_labels,
                             test_labels=dataset.test_labels)
    network.train(iteration_count=2000)
    [accuracy, predictions] = network.test()
    index, all_samples_v, correct_samples_v = 0, np.array([0.] * 10), np.array([0.] * 10)
    for prediction in predictions:
        all_samples_v[prediction] += 1
        if dataset.test_labels[index] == prediction:
            correct_samples_v[prediction] += 1
        index += 1
    print "Samples result vector: " + str(correct_samples_v / all_samples_v)
    print "Error rate for Backpropagation is: " + str(1 - accuracy)

    # Other classifier tests
    classifier = Classifier(dataset)
    classifiers = ['svm', 'rf', 'knn', 'llr']
    for c in classifiers:
        classifier.train(c)
        predictions = classifier.test()

        index, positive_count, all_samples = 0, 0, 0
        all_samples_v, correct_samples_v = np.array([0.] * 10), np.array([0.] * 10)
        for prediction in predictions:
            all_samples_v[prediction] += 1
            if dataset.test_labels[index] == prediction:
                positive_count += 1
                correct_samples_v[prediction] += 1
            all_samples += 1
            index += 1
        print "Error rate for " + str(c) + " is: " + str(1.0 - float(positive_count) / all_samples)
        print "Samples result vector: " + str(correct_samples_v / all_samples_v)


if __name__ == "__main__":
    _prepare_training_data()
    _run_classification()
    # reader = DatasetReader("Datasets")
    # data = reader.read_training_csv_set()
    # for i in range(18, 20):
    #    data[i].show(new_size=(300, 300))
