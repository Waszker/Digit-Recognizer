from image_tools.dataset_reader import DatasetReader

if __name__ == "__main__":
    reader = DatasetReader("Datasets")
    print 'Reading training data'
    data = reader.read_training_csv_set()
    print 'Finished reading training data'
    data[0].show(new_size=(300, 300))
    data[1].show(new_size=(300, 300))
    data[2].show(new_size=(300, 300))
