import numpy
from skimage.morphology import erosion, dilation, opening, closing, skeletonize, disk
from PIL import Image as PImage

default_image_size = (28, 28)


class Image:
    """
    Loads selected image and provides methods for its edition.
    """

    def __init__(self, pixel_values, image_dimensions=default_image_size, correct_class=None):
        self.correct_class = correct_class
        self.image_array = Image._read_image_array(pixel_values, image_dimensions)
        self.image_width = self.image_array.shape[0]
        self.image_height = self.image_array.shape[1]
        self.representative_vector = None

    def get_representative_vector(self):
        """
        Creates vector representation of image characteristics. To be used as input for classifier.
        :return: horizontal vector representation
        """
        if self.representative_vector is None:
            image = self.binarize()
            image = image.skeletonize()
            # vector = image.image_array.ravel()
            vector = list([self._get_black_pixels_count()])
            vector.append(image.count_starting_points())
            vector.append(image.count_intersection_points())
            self.representative_vector = numpy.asarray(vector)

        return self.representative_vector

    def show(self, new_size=default_image_size):
        # self.image_array = self.invert()
        img = self.binarize()
        img = img.skeletonize()
        img = PImage.fromarray(img.image_array)
        img = img.resize(new_size)
        # img = img.resize(new_size, PIL.Image.ANTIALIAS)
        # k3m.skeletize(img).show()
        img.show()

    def dilate(self, mask_size):
        """
        Performs dilation on binarized image.
        :param mask_size: size (width or height) of square mask used for dilation
        :return: dilated image
        """
        mask = disk(mask_size)
        img = dilation(self.image_array, mask)
        return Image(numpy.uint8(img), img.shape, self.correct_class)

    def erode(self, mask_size):
        """
        Performs erosion on binarized image.
        :param mask_size: size (width or height) of square mask used for erosion
        :return: eroded image
        """
        mask = disk(mask_size)
        img = erosion(self.image_array, mask)
        return Image(numpy.uint8(img), img.shape, self.correct_class)

    def skeletonize(self):
        """
        Converts shape in image to 1px wide one. Returns array of 8bit ints containing pixel color values.
        :return: skeletonized image
        """
        img = numpy.asarray(map(lambda x: x / 255, self.image_array))
        img = skeletonize(img)
        img = numpy.asarray(map(lambda x: x * 255, img))
        return Image(numpy.uint8(img), img.shape, self.correct_class)

    def invert(self):
        """
        Inverts grayscale image.
        :return: inverted image
        """
        img = numpy.asarray(map(lambda x: 255 - x, self.image_array))
        return Image(numpy.uint8(img), img.shape, self.correct_class)

    def binarize(self):
        """
        Converts image to black and white only.
        :return: binarized image
        """
        img = PImage.fromarray(self.image_array)
        img = img.convert('1')
        img = Image._read_image_array(numpy.array(img.getdata()), self.image_array.shape)

        return Image(numpy.uint8(img), img.shape, self.correct_class)

    def count_starting_points(self):
        """
        Calculates number of starting points for binary, skeletonized image.
        :return: number of starting points
        """
        starting_points = 0

        for i in range(0, self.image_width):
            for j in range(0, self.image_height):
                if self._is_binary_image_pixel_black(i, j) and self.get_pixel_neighbours(i, j) == 1:
                    starting_points += 1

        return starting_points

    def count_intersection_points(self):
        """
        Calculates number of intersection points for binary, skeletonized image.
        Based on article: https://arxiv.org/pdf/1202.3884.pdf
        :return: number of intersection points
        """
        intersection_points = 0

        for i in range(1, self.image_width - 1):
            for j in range(1, self.image_height - 1):
                neighbours_count = self.get_pixel_neighbours(i, j)
                if self._is_binary_image_pixel_black(i, j) and self._is_it_intersection_point(i, j, neighbours_count):
                    intersection_points += 1

        return intersection_points

    def get_pixel_neighbours(self, x, y):
        """
        Calculates number of pixel neighbours (black colored pixels around selected pixel.
        :param x: width coordinate of selected pixel
        :param y: height coordinate of selected pixel
        :return: number of black colored pixels (neighbours)
        """
        neighbours_count = 0

        if not self._are_coordinates_within_image(x, y):
            raise ValueError("X, Y values are not within image range!")

        for i in range(-1, 2):
            for j in range(-1, 2):
                if self._are_coordinates_within_image(x + i, y + j) \
                        and self._is_binary_image_pixel_black(x + i, y + j) and not (i == 0 and j == 0):
                    neighbours_count += 1

        return neighbours_count

    def _is_binary_image_pixel_black(self, x, y):
        black_values = [1, 255]
        return self.image_array[x][y] in black_values

    def _get_black_pixels_count(self):
        count = 0
        for i in range(1, self.image_width):
            for j in range(1, self.image_height):
                if self._is_binary_image_pixel_black(i, j):
                    count += 1
        return count

    def _are_coordinates_within_image(self, x, y):
        return not (x < 0 or x >= self.image_width or y < 0 or y >= self.image_height)

    def _is_it_intersection_point(self, i, j, neighbours_count):
        condition_1 = neighbours_count == 3 and self._is_intersection_with_three(i, j)
        condition_2 = neighbours_count == 4 and self._is_intersection_with_four(i, j)
        condition_3 = neighbours_count >= 5
        return condition_1 or condition_2 or condition_3

    def _is_intersection_with_three(self, i, j):
        top_left = self._is_binary_image_pixel_black(i - 1, j - 1) and (
            self._is_binary_image_pixel_black(i - 1, j) or self._is_binary_image_pixel_black(i, j - 1))
        top_center = self._is_binary_image_pixel_black(i, j - 1) and (
            self._is_binary_image_pixel_black(i - 1, j - 1) or self._is_binary_image_pixel_black(i + 1, j - 1))
        top_right = self._is_binary_image_pixel_black(i + 1, j - 1) and (
            self._is_binary_image_pixel_black(i, j - 1) or self._is_binary_image_pixel_black(i + 1, j))
        mid_left = self._is_binary_image_pixel_black(i - 1, j) and (
            self._is_binary_image_pixel_black(i - 1, j - 1) or self._is_binary_image_pixel_black(i - 1, j + 1))
        mid_right = self._is_binary_image_pixel_black(i + 1, j) and (
            self._is_binary_image_pixel_black(i + 1, j - 1) or self._is_binary_image_pixel_black(i + 1, j + 1))
        down_left = self._is_binary_image_pixel_black(i - 1, j + 1) and (
            self._is_binary_image_pixel_black(i - 1, j) or self._is_binary_image_pixel_black(i, j + 1))
        down_center = self._is_binary_image_pixel_black(i, j + 1) and (
            self._is_binary_image_pixel_black(i - 1, j + 1) or self._is_binary_image_pixel_black(i + 1, j + 1))
        down_right = self._is_binary_image_pixel_black(i + 1, j + 1) and (
            self._is_binary_image_pixel_black(i, j + 1) or self._is_binary_image_pixel_black(i + 1, j))

        answers = [top_left, top_center, top_right, mid_left, mid_right, down_left, down_center, down_right]

        return all(answers)

    def _is_intersection_with_four(self, i, j):
        return self._are_all_diagonal_intersections(i, j) and self._are_all_central_intersections(i, j)

    def _are_all_diagonal_intersections(self, i, j):
        top_left = self._is_binary_image_pixel_black(i - 1, j - 1) and (
            not (self._is_binary_image_pixel_black(i - 1, j) or self._is_binary_image_pixel_black(i, j - 1)))
        top_right = self._is_binary_image_pixel_black(i + 1, j - 1) and (
            not (self._is_binary_image_pixel_black(i, j - 1) or self._is_binary_image_pixel_black(i + 1, j)))
        down_left = self._is_binary_image_pixel_black(i - 1, j + 1) and (
            not (self._is_binary_image_pixel_black(i - 1, j) or self._is_binary_image_pixel_black(i, j + 1)))
        down_right = self._is_binary_image_pixel_black(i + 1, j + 1) and (
            not (self._is_binary_image_pixel_black(i, j + 1) or self._is_binary_image_pixel_black(i + 1, j)))

        answers = [top_left, top_right, down_left, down_right]

        return not all(answers)

    def _are_all_central_intersections(self, i, j):
        top_center = self._is_binary_image_pixel_black(i, j - 1) and (
            not (self._is_binary_image_pixel_black(i - 1, j - 1) or self._is_binary_image_pixel_black(i + 1, j - 1)))
        mid_left = self._is_binary_image_pixel_black(i - 1, j) and (
            not (self._is_binary_image_pixel_black(i - 1, j - 1) or self._is_binary_image_pixel_black(i - 1, j + 1)))
        mid_right = self._is_binary_image_pixel_black(i + 1, j) and (
            not (self._is_binary_image_pixel_black(i + 1, j - 1) or self._is_binary_image_pixel_black(i + 1, j + 1)))
        down_center = self._is_binary_image_pixel_black(i, j + 1) and (
            not (self._is_binary_image_pixel_black(i - 1, j + 1) or self._is_binary_image_pixel_black(i + 1, j + 1)))

        answers = [top_center, mid_left, mid_right, down_center]

        return not all(answers)

    @staticmethod
    def _read_image_array(pixel_values, image_dimensions):
        image_array = numpy.reshape(pixel_values, image_dimensions)
        return image_array
