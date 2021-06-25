import os
import cv2
from data import constants

def image_generator():
    files = os.listdir(constants.PADDED_IMAGE_DIR)

    for file in files:
        yield (cv2.imread(constants.PADDED_IMAGE_DIR + '/' + file) - 127.5) / 127.5