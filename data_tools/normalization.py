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

        return self._delete_zero_columns(np.asarray(dataset))

    @staticmethod
    def _get_min_max_vectors(image_list):
        pool = mp.Pool()
        manager = mp.Manager()
        dataset = manager.list()
        image_dataset = manager.list()
        for image in image_list:
            pool.apply_async(_append_image_vector, args=(dataset, image_dataset, image))
        pool.close()
        pool.join()
        dataset = np.asarray(dataset)
        del image_list[:]
        image_list.extend(image_dataset)

        min_vector = np.amin(dataset, axis=0)
        max_vector = np.amax(dataset, axis=0)

        return min_vector, max_vector

    def _delete_zero_columns(self, data):
        difference = np.tile(np.asarray(self.difference_vector), (2, 1))
        zero_columns = np.nonzero(difference.sum(axis=0) == 0)
        return np.delete(data, zero_columns, axis=1)


def _append_image_vector(dataset_list, image_dataset, image):
    dataset_list.append(image.get_representative_vector())
    image_dataset.append(image)


def _normalize_image(image, dataset, min_vector, difference_vector):
    row = image.get_representative_vector()
    row = np.asarray(row).astype(float)
    for i in range(0, len(difference_vector)):
        if difference_vector[i] == 0:
            row[i] = 0
        else:
            row[i] = float(row[i] - min_vector[i]) / difference_vector[i]
    dataset.append(row)
