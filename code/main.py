import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import PIL
from PIL import Image
from scipy import signal
from scipy import ndimage as ndi
#from ast import main

from skimage import (
    color, data, exposure, feature, filters, measure, morphology, segmentation, util
)
from skimage.filters import meijering, sato, frangi, hessian, threshold_otsu, gaussian, laplace, sobel, roberts
from skimage.measure import label, regionprops, regionprops_table
from skimage.segmentation import active_contour
from skimage.color import rgb2gray
from skimage.filters.thresholding import _cross_entropy

from vidstab import VidStab
import plotly
import plotly.express as px
import plotly.graph_objects as go
import BFEpreprocessing as bfe
import generate_mask as gm

def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x
    #return x.cuda(non_blocking=True) if torch.cuda.is_available() else x

def main():
    models_path = r'/home/stavb/robot-surgery-segmentation-master/data/models'
    video_path_1 = r'/home/stavb/PycharmProjects/BFE/Data/cut2.mov'
    #TODO: here should be the part where we load the video
    #vid = bfe.load_video(video_path_1)

    # temporary part #
    image_path = r'/home/stavb/PycharmProjects/BFE/Data/frames'
    image_format = "png"
    images=[]
    for root, dirs, files in os.walk(image_path):
        for filename in files:
            if filename.lower().endswith(image_format):
                images.append(os.path.join(root, filename))

    #for img in images:  # 1 img for now
    for img in [images[0]]:
        for root, dirs, files in os.walk(models_path):
            for filename in files:
                model_path = os.path.join(root, filename)
                model_type = (root.split("/")[-1]).split("_")[0]
                model_name = model_type + filename
                mask_array=gm.generate_mask(model_path, img, model_type, show=0)
                imshow(gm.mask_overlay(image, (mask_array > 0).astype(np.uint8)))

    pass


if __name__ == '__main__':
    main()



