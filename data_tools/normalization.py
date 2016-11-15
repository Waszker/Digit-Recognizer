import numpy as np
import multiprocessing as mp


class Normalizer:
    """
    Class responsible for normalizing image datasets and storing values for future normalizations.
    """

    def __init__(self, image_list):
        self.min_vector, self.max_vector = Normalizer._get_min_max_vectors(image_list)
        self.difference_vector = self.max_vector - self.min_vector

    def get_normalized_data_matrix(self, image_list):
        """
        Normalizes provided list of images' vectors.
        :param image_list: list of images convert into input matrix
        :return: normalized input matrix
        """
        manager = mp.Manager()
        dataset = manager.list()
        pool = mp.Pool()
        for image in image_list:
            pool.apply_async(_normalize_image, args=(image, dataset, self.min_vector, self.difference_vector))
        pool.close()
        pool.join()

        return np.asarray(dataset)

    @staticmethod
    def _get_min_max_vectors(image_list):
        pool = mp.Pool()
        manager = mp.Manager()
        dataset = manager.list()
        for image in image_list:
            pool.apply_async(_append_image_vector, args=(dataset, image))
        pool.close()
        pool.join()
        dataset = np.asarray(dataset)

        min_vector = np.amin(dataset, axis=0)
        max_vector = np.amax(dataset, axis=0)

        return min_vector, max_vector


def _append_image_vector(dataset_list, image):
    dataset_list.append(image.get_representative_vector())


def _normalize_image(image, dataset, min_vector, difference_vector):
    row = image.get_representative_vector()
    for i in range(0, len(difference_vector)):
        if difference_vector[i] == 0:
            row[i] = 0
        else:
            row[i] = (row[i] - min_vector[i]) / difference_vector[i]
    dataset.append(row)
