from data_tools.dataset_reader import DatasetReader
from data_tools.normalization import Normalizer
from classifiers.classifier import Classifier

if __name__ == "__main__":
    reader = DatasetReader("Datasets")
    print 'Reading training data'
    data = reader.read_training_csv_set()
    print 'Finished reading training data'
    normalizer = Normalizer(data)
    normalized_data = normalizer.get_normalized_data_matrix(data)
    data[0].show(new_size=(300, 300))
    data[1].show(new_size=(300, 300))
    data[2].show(new_size=(300, 300))
