""" Image-related utility functions """
# TODO: Add testing for this module

import math
from enum import Enum, auto
from random import randint
from typing import Tuple

# Try to import numpy and PIL, but ignore errors to avoid forcing the dependency
try:
    import numpy
    from numpy import ndarray as NDArrayType
except ImportError:
    from typing import Any as NDArrayType

try:
    from PIL import Image, ImageOps
    from PIL import Image as ImageType
except ImportError:
    from typing import Any as ImageType


def normalize(
    img: NDArrayType, means: Tuple[float] = None, stdevs: Tuple[float] = None
) -> NDArrayType:
    """ Normalize an image by performing `(array - mean) / stdev` to each channel. """
    if means is None:
        means = [arr.mean() for arr in img]
    if stdevs is None:
        stdevs = [arr.std() for arr in img]
    assert img.ndim == 3, "normalize function requires 3 channels in image, found %r" % img.shape
    return numpy.asarray(
        [(arr - means[i]) / stdevs[i] for i, arr in enumerate(img)], dtype=numpy.float64
    )


def image_to_ndarray(img: ImageType) -> NDArrayType:
    """ Converts a PIL.Image type to numpy.ndarray. """
    # Convert into ndarray type
    arr = numpy.array(img).astype(numpy.uint8)

    # If image is grayscale, we are done
    if arr.ndim == 2:
        return arr

    # Otherwise we need to change the shape to put the RGB planes first
    return numpy.array([arr[:, :, i] for i in range(arr.shape[2])])


def ndarray_to_image(img: NDArrayType) -> ImageType:
    """ Converts a numpy.ndarray type to a PIL.Image type. """

    assert img.dtype == numpy.uint8, "Array must be of uint8 dtype"

    # Early exit: 2D array does not need to take care of order of planes
    if img.ndim == 2:
        return Image.fromarray(img)

    # Otherwise we need to put the RGB planes first, then height and width
    img_ = numpy.zeros((img.shape[1], img.shape[2], img.shape[0]), dtype=numpy.uint8)
    for channel in range(img.shape[0]):
        img_[:, :, channel] = img[channel, :, :]

    return Image.fromarray(img_)


def _exif_rotate(image: ImageType) -> ImageType:
    """
    https://github.com/python-pillow/Pillow/issues/4346#issuecomment-575049324
    """

    # Extract image EXIF data
    exif = image.getexif()

    # Remove all EXIF tags except rotation information to work around PIL bugs
    for k in exif.keys():
        if k != 0x0112:
            exif[k] = None
            del exif[k]

    # Put the new EXIF object in the original image
    new_exif = exif.tobytes()
    image.info["exif"] = new_exif

    # Rotate the image and return
    return ImageOps.exif_transpose(image)


def open_image(
    path: str, convert: str = None, channels: bool = False, uint8: bool = True
) -> NDArrayType:
    """ Opens an image given its path. Fails if PIL package is not installed. """

    # Open the image using PIL
    img = Image.open(path)
    img = _exif_rotate(img)

    # Convert to a different colorspace if needed
    if convert:
        img_ = img.convert(convert)
        img.close()
        img = img_

    # Convert image to numpy array
    img_ = image_to_ndarray(img)
    img.close()
    img = img_

    # Make sure that we have explicit image channels
    if channels and img.ndim == 2:
        img = img.reshape((1, *img.shape))

    # Convert to unsigned type if required
    if not uint8:
        img = img.astype(float) / 255

    return img


def _get_image_size(img: NDArrayType):
    """ Returns the width and height of the image for the 2D and 3D case """
    if img.ndim == 2:
        return tuple(img.shape)
    if img.ndim == 3:
        return tuple(img.shape[1:])
    raise ValueError("Input image has more than 3 dimensions. Shape found: %r" % img.shape)


def _set_image_pixel(img: NDArrayType, x: int, y: int, value: int):
    if img.ndim == 2:
        img[y, x] = value
        return
    if img.ndim == 3:
        img[:, y, x] = value
        return
    raise ValueError("Input image has more than 3 dimensions. Shape found: %r" % img.shape)


def _get_image_pixel(img: NDArrayType, x: int, y: int):
    if img.ndim == 2:
        return img[y, x]
    if img.ndim == 3:
        return img[:, y, x]
    raise ValueError("Input image has more than 3 dimensions. Shape found: %r" % img.shape)


def _get_image_region(img: NDArrayType, x0: int, x1: int, y0: int, y1: int):
    if img.ndim == 2:
        return img[y0:y1, x0:x1]
    if img.ndim == 3:
        return img[:, y0:y1, x0:x1]
    raise ValueError("Input image has more than 3 dimensions. Shape found: %r" % img.shape)


class CropStrategy(Enum):
    """ Image cropping algorithm. """

    TOP_LEFT = auto()
    CENTER = auto()
    RANDOM = auto()


def _crop_pixel_count(img_height: int, img_width: int, crop_height: int, crop_width: int):
    assert crop_height <= img_height and crop_width <= img_width, (
        "Cropping region (%d x %d) must be smaller than image size (%d x %d)"
        % (crop_height, crop_width, img_height, img_width)
    )

    return img_height - crop_height, img_width - crop_width


def _compute_crop_region(
    method: CropStrategy, img_height: int, img_width: int, crop_height: int, crop_width: int
):
    crop_y, crop_x = _crop_pixel_count(img_height, img_width, crop_height, crop_width)

    if method == CropStrategy.TOP_LEFT:
        return (0, 0, crop_height, crop_width)

    if method == CropStrategy.CENTER:
        crop_y_half = crop_y // 2
        crop_x_half = crop_x // 2
        return (crop_y_half, crop_x_half, crop_y_half + crop_height, crop_x_half + crop_width)

    if method == CropStrategy.RANDOM:
        crop_y_0 = randint(0, crop_y)
        crop_x_0 = randint(0, crop_x)
        crop_y_1 = crop_y - crop_y_0
        crop_x_1 = crop_x - crop_x_0
        return (crop_y_0, crop_x_0, crop_y_1, crop_x_1)


def crop(img: NDArrayType, height: int, width: int, method: CropStrategy = CropStrategy.CENTER):
    """
    Crops an image using the given strategy. Image is automatically resized if necessary.
    """
    img_h, img_w = _get_image_size(img)

    # If the image is smaller than desired crop region, scale it up preserving aspect ratio
    scale_aspect_ratio = max(height / float(img_h), width / float(img_w))
    assert scale_aspect_ratio <= 1, "Crop region is larger than image size"
    # if scale_aspect_ratio > 1:
    #     img = scale(img, round(img_h * scale_aspect_ratio), round(img_w * scale_aspect_ratio))
    #     img_h, img_w = _get_image_size(img)

    y0, x0, y1, x1 = _compute_crop_region(method, img_h, img_w, height, width)
    return _get_image_region(img, x0, x1, y0, y1)


class ScalingMethod(Enum):
    """ Image scaling algorithm """

    NEAREST_NEIGHBOR = auto()


def _scale_nearest_neighbor(img: NDArrayType, height: int, width: int):
    img_out = None
    if img.ndim == 2:
        img_out = numpy.zeros((height, width), dtype=img.dtype)
    if img.ndim == 3:
        img_out = numpy.zeros((img.shape[0], height, width), dtype=img.dtype)

    img_h, img_w = _get_image_size(img)
    for y in range(0, height):
        for x in range(0, width):
            src_y = min(img_h - 1, round(float(y) / float(height) * float(img_h)))
            src_x = min(img_w - 1, round(float(x) / float(width) * float(img_w)))
            _set_image_pixel(img_out, x, y, _get_image_pixel(img, src_y, src_x))
    return img_out


def scale(
    img: NDArrayType,
    height: int,
    width: int,
    method: ScalingMethod = ScalingMethod.NEAREST_NEIGHBOR,
) -> NDArrayType:
    """ Scales an image using the provided method, e.g. nearest neighbor. """

    if method == ScalingMethod.NEAREST_NEIGHBOR:
        return _scale_nearest_neighbor(img, height=height, width=width)
    else:
        raise ValueError("Unknown scaling method requested: %r" % method)


def rotate(
    img: NDArrayType, rotation_degrees: int, fill_color_rgb: Tuple[int, int, int] = (255,) * 3
) -> NDArrayType:
    """ Rotates an image by `rotation_degrees` and fills the corners with `fill_color_rgb` """

    # Original image in PIL.Image format
    im_orig = ndarray_to_image(img)
    # Filled image the same size as the original image
    im_filled_orig = Image.new("RGBA", im_orig.size, tuple([*fill_color_rgb, 255]))
    # Filled image with rotation applied
    im_filled_rotated = im_filled_orig.rotate(rotation_degrees)
    # Filled image same size as post-rotated image
    im_filled_post = Image.new("RGBA", im_filled_rotated.size, tuple([*fill_color_rgb, 255]))
    # Create a composite image using the alpha layer of im_white_rotated as a mask
    im_output = Image.composite(im_orig.rotate(rotation_degrees), im_filled_post, im_filled_rotated)
    # Output image with same color mode as original
    return image_to_ndarray(im_output.convert(im_orig.mode))


def rgb_to_grayscale(img: NDArrayType) -> NDArrayType:
    """ Converts an image with 3 channels from RGB to grayscale color space """
    assert img.ndim == 3, "Image expected to have 3 channels for RGB"
    return img[:, :, 0] * 0.2126 + img[:, :, 0] * 0.7152 + img[:, :, 0] * 0.0722


def resize_canvas(
    img: NDArrayType, height: int, width: int, fill_color_rgb: Tuple[int, int, int] = (255,) * 3
) -> NDArrayType:
    """ Resizes the canvas of an image, cropping the image if necessary """
    if height < img.shape[0]:
        img = crop(img, height, img.shape[1], CropStrategy.TOP_LEFT)
    if width < img.shape[1]:
        img = crop(img, img.shape[0], width, CropStrategy.TOP_LEFT)

    rgb = img.ndim == 3
    fill_color = (
        fill_color_rgb
        if rgb
        else math.ceil(rgb_to_grayscale(numpy.array(fill_color_rgb).reshape(1, 1, 3)))
    )
    canvas_shape = (height, width, 3) if rgb else (height, width)
    img_new = numpy.ones(canvas_shape, dtype=img.dtype) * fill_color

    img_height, img_width = img.shape[:2]
    y_delta = (height - img_height) // 2
    x_delta = (width - img_width) // 2
    img_new[y_delta : y_delta + img_height, x_delta : x_delta + img_width] = img
    return img_new
