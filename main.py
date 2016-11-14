from data_tools.dataset_reader import DatasetReader
from data_tools.normalization import Normalizer

if __name__ == "__main__":
    reader = DatasetReader("Datasets")
    print 'Reading training data'
    data = reader.read_test_csv_set()
    print 'Finished reading training data'
    normalizer = Normalizer(data)
    print 'Finished normalizing training data'
    normalized_data = normalizer.get_normalized_data_matrix(data)
    print normalized_data.shape
    data[0].show(new_size=(300, 300))
    data[1].show(new_size=(300, 300))
    data[2].show(new_size=(300, 300))
