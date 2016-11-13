import numpy
from skimage.morphology import erosion, dilation, opening, closing, white_tophat
from skimage.morphology import black_tophat, skeletonize, convex_hull_image
from skimage.morphology import disk
from PIL import Image as PImage

default_image_size = (28, 28)


class Image:
    """
    Loads selected image and provides methods for its edition.
    """

    def __init__(self, pixel_values, image_dimensions=default_image_size, correct_class=None):
        self.correct_class = correct_class
        self.image_array = Image._read_image_array(pixel_values, image_dimensions)

    def show(self, new_size=default_image_size):
        # self.image_array = self.invert()
        self.image_array = self.binarize()
        self.image_array = self.skeletonize()
        print "Starting points: " + str(self.count_starting_points())
        img = PImage.fromarray(self.image_array)
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
        return numpy.uint8(img)

    def erode(self, mask_size):
        """
        Performs erosion on binarized image.
        :param mask_size: size (width or height) of square mask used for erosion
        :return: eroded image
        """
        mask = disk(mask_size)
        img = erosion(self.image_array, mask)
        return numpy.uint8(img)

    def skeletonize(self):
        """
        Converts shape in image to 1px wide one. Returns array of 8bit ints containing pixel color values.
        :return: skeletonized image
        """
        img = numpy.asarray(map(lambda x: x / 255, self.image_array))
        img = skeletonize(img)
        img = numpy.asarray(map(lambda x: x * 255, img))
        return numpy.uint8(img)

    def invert(self):
        """
        Inverts grayscale image.
        :return: inverted image
        """
        return numpy.asarray(map(lambda x: 255 - x, self.image_array))

    def binarize(self):
        """
        Converts image to black and white only.
        :return: binarized image
        """
        img = PImage.fromarray(self.image_array)
        img = img.convert('1')
        img = Image._read_image_array(numpy.array(img.getdata()), self.image_array.shape)

        return numpy.uint8(img)

    def count_starting_points(self):
        """
        Calculates number of starting points for binary, skeletonized image.
        :return: number of starting points
        """
        starting_points = 0

        for i in range(0, self.image_array.shape[0]):
            for j in range(0, self.image_array.shape[1]):
                if self._is_binary_image_pixel_black(i, j) and self.get_pixel_neighbours(i, j) == 1:
                    starting_points += 1

        return starting_points

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

    def _are_coordinates_within_image(self, x, y):
        image_width = self.image_array.shape[0]
        image_height = self.image_array.shape[1]
        return not (x < 0 or x >= image_width or y < 0 or y >= image_height)

    @staticmethod
    def _read_image_array(pixel_values, image_dimensions):
        image_array = numpy.reshape(pixel_values, image_dimensions)
        return image_array
