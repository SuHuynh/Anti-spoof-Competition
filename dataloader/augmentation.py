from imgaug import augmenters as iaa
import math
import cv2
import numpy as np
from PIL import Image
# from data_helper import *

def random_cropping(image, target_shape=(32, 32, 3), is_random = True):

    height, width, _ = image.shape

    if is_random:
        start_x = random.randint(0, int(0.05*width))
        start_y = random.randint(0, int(0.05*height))

    new_img = image[start_y:height - start_y,start_x:width - start_x,:]
    return new_img

def color_augumentor(image):

    image = np.array(image)

    augment_img = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.2),
        iaa.Affine(rotate=(-30, 30)),
    ], random_order=True)

    image = augment_img.augment_image(image)
    # image = random_resize(image)
    # image = random_cropping(image, target_shape, is_random=True)
    image = Image.fromarray(image)
    return image

def depth_augumentor(image, target_shape=(32, 32, 3), is_infer=False):
    if is_infer:
        augment_img = iaa.Sequential([
            iaa.Fliplr(0),
        ])

        image =  augment_img.augment_image(image)
        image = TTA_36_cropps(image, target_shape)
        return image

    else:
        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(rotate=(-30, 30)),
        ], random_order=True)

        image = augment_img.augment_image(image)
        image = random_resize(image)
        image = random_cropping(image, target_shape, is_random=True)
        return image

def ir_augumentor(image, target_shape=(32, 32, 3), is_infer=False):
    if is_infer:
        augment_img = iaa.Sequential([
            iaa.Fliplr(0),
        ])
        image =  augment_img.augment_image(image)
        image = TTA_36_cropps(image, target_shape)
        return image

    else:
        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(rotate=(-30, 30)),
        ], random_order=True)

        image = augment_img.augment_image(image)
        image = random_resize(image)
        image = random_cropping(image, target_shape, is_random=True)
        return image