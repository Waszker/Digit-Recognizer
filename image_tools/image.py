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
        img = PImage.fromarray(self.image_array)
        img = img.resize(new_size)
        # img = img.resize(new_size, PIL.Image.ANTIALIAS)
        # k3m.skeletize(img).show()
        img.show()

    def dilate(self, mask_size):
        mask = disk(mask_size)
        img = dilation(self.image_array, mask)
        return numpy.uint8(img)

    def erode(self, mask_size):
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

    @staticmethod
    def _read_image_array(pixel_values, image_dimensions):
        image_array = numpy.reshape(pixel_values, image_dimensions)
        return image_array
