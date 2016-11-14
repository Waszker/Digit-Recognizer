import numpy as np

class Normalizer:
    """
    Class responsible for normalizing image datasets and storing values for future normalizations.
    """
    def __init__(self, image_list):
        self.min_vector, self.max_vector = self._get_min_max_vectors(image_list)
        self.difference_vector = self.max_vector - self.min_vector

    def get_normalized_data_matrix(self, image_list):
        """
        TODO: Finish description
        :param image_list:
        :return:
        """
        dataset = []
        for image in image_list:
            row = image.get_representative_vector()
            for i in range(0, len(self.difference_vector)):
                row[i] = (row[i] - self.min_vector[i]) / self.difference_vector[i]
            dataset.append(row)

        return np.asarray(dataset)

    def _get_min_max_vectors(self, image_list):
        dataset = []
        for image in image_list:
            dataset.append(image.get_representative_vector())
        dataset = np.asarray(dataset)

        min_vector =  np.amin(dataset, axis = 0)
        max_vector =  np.amax(dataset, axis = 0)

        return min_vector, max_vector
