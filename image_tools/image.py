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
        self.image_width = self.image_array.shape[1]
        self.image_height = self.image_array.shape[0]
        self.representative_vector = None

    def get_representative_vector(self):
        """
        Creates vector representation of image characteristics. To be used as input for classifier.
        :return: horizontal vector representation
        """
        if self.representative_vector is None:
            # image = self.binarize()
            # image = image.skeletonize()
            vector = self.image_array.ravel().tolist()
            # vector = list([self._get_black_pixels_count()])
            # vector.extend(self._shrink_image_array())
            # vector.append(image.count_starting_points())
            # vector.append(image.count_intersection_points())
            # vector.extend(image.get_intersections_vector())
            self.representative_vector = numpy.asarray(vector)

        return self.representative_vector

    def show(self, new_size=default_image_size):
        # self.image_array = self.invert()
        img = self.binarize()
        img = img.dilate(2)
        img = img.skeletonize()
        print 'Starting points ' + str(img.count_starting_points())
        print 'Intersection points ' + str(img.count_intersection_points())
        print img.get_intersections_vector()
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

    def get_intersections_vector(self):
        """
        Calculates number of intersections with line at certain width and height values on binary image.
        Algorithm counts intersections with line at 30%, 50% and 70% of image width/height.
        :return: vector containing number of intersections at 30%, 50% and 70% of image width and height
        """
        width_values = self._get_symbol_starting_and_ending_width()
        height_values = self._get_symbol_starting_and_ending_height()

        width_step = (float(width_values[1] - width_values[0])) / 10
        height_step = (float(height_values[1] - height_values[0])) / 10

        step_coefficients = [3, 5, 7]  # 30%, 50%, 70% of image width/height
        intersection_vector = []

        for coefficient in step_coefficients:
            intersection_vector.append(
                self._get_intersection_count_heightwise(width_values[0] + int(coefficient * width_step)))
        for coefficient in step_coefficients:
            intersection_vector.append(
                self._get_intersection_count_widewise(height_values[0] + int(coefficient * height_step)))

        return intersection_vector

    def _is_binary_image_pixel_black(self, x, y):
        black_values = [1, 255]
        return self.image_array[y][x] in black_values

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

    def _shrink_image_array(self, mask_size=(4, 4)):
        if not (self.image_width % mask_size[0] == 0 or self.image_height % mask_size[1] == 0):
            raise ValueError("Mask size must fit within image dimensions")

        shrinked = []
        for i in range(0, self.image_width, mask_size[0]):
            for j in range(0, self.image_height, mask_size[1]):
                value = 0.
                for ii in range(0, mask_size[0]):
                    for jj in range(0, mask_size[1]):
                        value += self.image_array[i + ii][j + jj]
                value /= mask_size[0] * mask_size[1]
                shrinked.append(value)

        return shrinked

    def _get_symbol_starting_and_ending_width(self):
        starting_width, ending_width = 0, 0

        try:
            for x in range(0, self.image_width):
                for y in range(0, self.image_height):
                    if self._is_binary_image_pixel_black(x, y):
                        starting_width = x
                        raise ValueError
        except ValueError:
            pass

        try:
            for x in range(self.image_width - 1, starting_width, -1):
                for y in range(0, self.image_height):
                    if self._is_binary_image_pixel_black(x, y):
                        ending_width = x
                        raise ValueError
        except ValueError:
            pass

        return starting_width, ending_width

    def _get_symbol_starting_and_ending_height(self):
        starting_height, ending_height = 0, 0

        try:
            for y in range(0, self.image_height):
                for x in range(0, self.image_width):
                    if self._is_binary_image_pixel_black(x, y):
                        starting_height = y
                        raise ValueError
        except ValueError:
            pass

        try:
            for y in range(self.image_height - 1, starting_height, -1):
                for x in range(0, self.image_width):
                    if self._is_binary_image_pixel_black(x, y):
                        ending_height = y
                        raise ValueError
        except ValueError:
            pass

        return starting_height, ending_height

    def _get_intersection_count_widewise(self, line_y):
        intersections = 0
        was_previously_black = False

        for i in range(0, self.image_width):
            if self._is_binary_image_pixel_black(i, line_y) and not was_previously_black:
                intersections += 1
                was_previously_black = True
            elif not self._is_binary_image_pixel_black(i, line_y):
                was_previously_black = False

        return intersections

    def _get_intersection_count_heightwise(self, line_x):
        intersections = 0
        was_previously_black = False

        for i in range(0, self.image_height):
            if self._is_binary_image_pixel_black(line_x, i) and not was_previously_black:
                intersections += 1
                was_previously_black = True
            elif not self._is_binary_image_pixel_black(line_x, i):
                was_previously_black = False

        return intersections

    @staticmethod
    def _read_image_array(pixel_values, image_dimensions):
        image_array = numpy.reshape(pixel_values, image_dimensions)
        return image_array
