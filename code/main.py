import os
import matplotlib.pyplot as plt
from plotWindow import plotWindow
import random

import numpy as np
import cv2
from cv2 import imshow
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

def show_models(images,models_path):
    # shows all the different model for surgical tool segmentation
    fig_num=0
    pw = plotWindow()
    for img in images:
        rows = 4
        columns = 4
        #fig = plt.figure(num=fig_num,figsize=(30, 30))
        fig = plt.figure(figsize=(30, 30))
        fig_num=fig_num+1

        #plt.title(img+'\n',fontsize=14, fontweight='bold')
        i=1
        for root, dirs, files in os.walk(models_path):
            for filename in files:
                model_path = os.path.join(root, filename)
                model_type = (root.split("/")[-1]).split("_")[0]
                model_name = model_type +" "+ filename
                mask_array,mask_overlayed=gm.generate_mask(model_path, img, model_type, show=0, cut=[576,704])

                # fig.add_subplot(rows, columns, i)
                # plt.imshow(mask_array>0)
                # plt.axis('off')
                # plt.title("mask_array"+model_name)

                fig.add_subplot(rows, columns, i)
                plt.imshow(mask_overlayed)
                plt.axis('off')
                plt.title(model_name)

                i=i+1
        pw.addPlot(str(fig_num), fig)

def show_images(images,best_model_path,best_model_type,labels=0,nums=0):
    colors = ['red', 'green', 'blue']
    hsv_list = ['hue', 'saturation', 'value']
    pw2 = plotWindow()
    rows = 4
    columns = 4
    for count,img in enumerate(images):
    #     # for img in [images[random.randint(0, len(images)-1)]]:
        OG_img = gm.load_image(img)
        fig = plt.figure(num=count+1)
        fig.add_subplot(rows, columns, 1)
        plt.imshow(OG_img)
        plt.axis('off')
        plt.title("Original Image")
        for i in range(3):
            fig.add_subplot(rows, columns, i + 2)
            tmp = np.zeros(OG_img.shape, dtype='uint8')
            tmp[:, :, i] = OG_img[:, :, i]
            plt.imshow(tmp)
            plt.axis('off')
            plt.title(colors[i])

        img_hsv = cv2.cvtColor(OG_img, cv2.COLOR_RGB2HSV)
        # hsv_out = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        for i in range(3):
            fig.add_subplot(rows, columns, i + 6)
            plt.imshow(img_hsv[:, :, i], cmap='gray', vmin=0, vmax=255)
            plt.axis('off')
            plt.title(hsv_list[i])
        mask_array = gm.generate_mask(best_model_path, img, best_model_type, show=0, cut=[576, 704])
        mask=(mask_array>0).astype(np.uint8)

        fig.add_subplot(rows, columns, 9)
        plt.imshow(mask)
        plt.axis('off')
        plt.title("original mask")

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_mask = np.ones(mask.shape[:2], dtype="uint8") * 255
        for c in contours:
            if c.size < 150:
                cv2.drawContours(contour_mask, [c], -1, 0, -1)
        mask2 = cv2.bitwise_and(mask, mask, mask=contour_mask)
        fig.add_subplot(rows, columns, 10)
        plt.imshow(mask2)
        plt.axis('off')
        plt.title("new mask")
        mask_overlayed = gm.mask_overlay(OG_img, mask2)
        fig.add_subplot(rows, columns, 11)
        plt.imshow(mask_overlayed)
        plt.title('with mask')
        plt.axis('off')

        fig.add_subplot(rows, columns, 13)
        bfe.colon_seg(img, fig.axes[-1])
        plt.title("active contour")

        fig.add_subplot(rows, columns, 14)
        frame = bfe.load_frame(img)
        plt.imshow(frame)
        plt.axis('off')
        plt.title("equalized green")
        binary_frame = bfe.threshold(bfe.F_frangi(frame))
        fig.add_subplot(rows, columns, 15)
        plt.imshow(binary_frame)
        plt.axis('off')
        plt.title("vessels")
        fig.add_subplot(rows, columns, 16)
        bfe.find_vessels(binary_frame)
        if labels:
            fig.add_subplot(rows, columns, 5)
            plt.imshow(overlay(img,labels[count]))
            plt.axis('off')
            plt.title(nums[count])
        plt.tight_layout()

        pw2.addPlot(str(count+1), fig)
        print(count)


    pw2.show()

def overlay(img_path,label_path):
    img = gm.load_image(img_path)
    label = (gm.load_image(label_path).astype(np.uint8))
    weighted_sum = cv2.addWeighted(label, 0.5, img, 0.5, 0.)
    image = img.copy()
    image = image[:label.shape[0], :label.shape[1], :]
    ind = ((label[:, :, 0] + label[:, :, 1]) > 0)
    image[ind] = weighted_sum[ind]
    return image


def main():
    models_path = r'/home/stavb/robot-surgery-segmentation-master/data/models'
    video_path_1 = r'/home/stavb/PycharmProjects/BFE/Data/cut2.mov'
    #TODO: here should be the part where we load the video
    #vid = bfe.load_video(video_path_1)

    # temporary part #
    #add all images paths to 'images' list
    image_path = r'/home/stavb/PycharmProjects/BFE/Data/frames'
    image_format = "png"
    images=[]
    for root, dirs, files in os.walk(image_path):
        for filename in files:
            if filename.lower().endswith(image_format):
                images.append(os.path.join(root, filename))

    #show_models(images,models_path)
    best_model_path = os.path.join(models_path, 'unet16_binary_20', 'model_1.pt')
    best_model_type = 'UNet16'
    #show_images(images, best_model_path, best_model_type)

    image_path = r'/home/stavb/frames_im'
    image_format = "png"
    images = []
    labels=[]
    nums=[]
    for root, dirs, files in os.walk(image_path):
        for filename in files:
            if filename.lower().endswith(image_format):
                if filename == 'img.png':
                    images.append(os.path.join(root, filename))
                elif filename == 'label.png':
                    labels.append(os.path.join(root, filename))
                    nums.append((root.split('/')[4]).split("_")[1])
    show_images(images, best_model_path, best_model_type,labels =labels,nums=nums)

    pass


if __name__ == '__main__':
    main()



