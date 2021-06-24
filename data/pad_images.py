import os
import cv2
import numpy as np
import constants

from PIL import Image

def get_max_image_sizes(files):
    max_width = 0
    max_height = 0
    
    for i, file in enumerate(files):
        # print("{}/{}".format(i+1, len(files)), end='\r')
        im = Image.open(constants.IMAGES_DIR + '/' + file, 'r')
        max_height = max(im.height, max_width)
        max_width = max(im.width, max_width)

    return max_width, max_height

def pad():
    files = os.listdir(constants.IMAGES_DIR)
    max_width, max_height = get_max_image_sizes(files)

    for i, file in enumerate(files):
        print("padding image {}/{}".format(i+1, len(files)), end='\r')
        img = cv2.imread(constants.IMAGES_DIR + '/' + file)
        img_hight, image_width, color_chanels = img.shape

        # create new image of desired size and color (blue) for padding
        # ww = 300
        # hh = 300
        # color = (255,0,0)
        result = np.full((max_height,max_width, color_chanels), constants.PADDING_COLOR, dtype=np.uint8)

        # compute center offset
        xx = (max_width - image_width) // 2
        yy = (max_height - img_hight) // 2

        # copy img image into center of result image
        result[yy:yy+img_hight, xx:xx+image_width] = img

        # # view result
        # cv2.imshow("result", result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # save result
        cv2.imwrite(constants.PADDED_IMAGE_DIR + '/' + file, result)

    print('\n')

if __name__ == '__main__':
    pad()
    