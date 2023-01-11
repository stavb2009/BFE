import os
import matplotlib.pyplot as plt
from plotWindow import plotWindow
import random
import torch
import numpy as np
import cv2
from cv2 import imshow
import PIL
from PIL import Image
from scipy import signal
from scipy import ndimage as ndi
#from ast import main
import temp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from skimage import (
    color, data, exposure, feature, filters, measure, morphology, segmentation, util
)
from skimage.filters import meijering, sato, frangi, hessian, threshold_otsu, gaussian, laplace, sobel, roberts
from skimage.measure import label, regionprops, regionprops_table
from skimage.segmentation import active_contour
from skimage.color import rgb2gray
from skimage.filters.thresholding import _cross_entropy
import pickle

from vidstab import VidStab
import plotly
import plotly.express as px590
import plotly.graph_objects as go
import BFEpreprocessing as bfe
import generate_mask as gm
import utils
import classify
import pandas as pd
import ofirs

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
                mask_array=gm.generate_mask(model_path, img, model_type, show=0, cut=[576,704])
                mask = (mask_array > 0).astype(np.uint8)
                mask_overlayed = gm.mask_overlay(gm.load_image(img), mask)
                # fig.add_subplot(rows, columns, i)
                # plt.imshow(mask_array>0)
                # plt.axis('off')
                # plt.title("mask_array"590+model_name)

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
        # hsv_out = cv2.cvtColor(hsv_ima590ge, cv2.COLOR_HSV2BGR)
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

def crop_bad(img):
    # crop video ro contain area with bad blood flow
    left = 259070
    top = 150
    right = 520
    bottom = 200
    bad = img[top:bottom, left:right,:]
    return bad

def crop_good(img):
    # crop video ro contain area with good blood flow
    left = 40
    top = 110
    right = 110
    bottom = 390
    good = img[top:bottom, left:right,:]
    return good

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
    vid_path = r'/home/stavb/ch1_video_01.mpg'

    # temporary part #
    #add all images paths to 'images' list

    new_frames_path = r'/home/stavb/new_frames'
    path_good = os.path.join(new_frames_path, 'good')
    path_poor = os.path.join(new_frames_path, 'poor')
    full_vid_path = r'/home/stavb/ch1_video_01.mpg'
    # labels = ["Good","Poor"]
    # frames = {}
    # image_format = "jpg"
    #
    # for label_tag,label in enumerate(labels):
    #     path_to_frames = os.path.join(new_frames_path,label)
    #     for root, dirs, files in os.walk(path_to_frames):
    #         for filename in files:
    #             if filename.lower().endswith(image_format):
    #                 frame_num = filename.split('.')[0]
    #                 frame_path = os.path.join(root, filename)
    #                 frames[frame_num] = {}
    #                 frames[frame_num]['path'] = frame_path
    #                 frames[frame_num]['label'] =label
    #
    #
    # for frame_num in frames.keys():
    #     #vid, full_color, fs = ofirs.load_video(full_vid_path, start = int(frame_num), duration = 5)
    #     #image = cv2.imread(frames[frame_num]['path'])
    #     #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     #TODO: apply mask
    #     # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #     out_path= os.path.join(new_frames_path,frames[frame_num]['label']+"_vid",frame_num+'.avi')
    #     # shape_out = (full_color.shape[2],full_color.shape[1])
    #     # out = cv2.VideoWriter(out_path, fourcc, float(fs), shape_out)
    #     # for frame in full_color:
    #     #     # Convert the frame to a proper image format
    #     #     #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #     #     # Write the frame to the video file
    #     #     out.write(frame.astype('uint8'))
    #     # out.release()
    #     out_path_stab = os.path.join(new_frames_path, frames[frame_num]['label'] + "_vid", frame_num + '_stab.avi')
    #     #bfe.video_stab(out_path,out_path_stab)
    #     frames[frame_num]['vid'] = out_path
    #     frames[frame_num]['stab'] = out_path_stab
    #
    #
    #     pass
    # #full_color,borders = utils.img2vid()
    # #data_from_vid = ofirs.time_corp_analyze_shell(full_color, stride=0, dur=5, borders = borders, label='good',
    # #                              fs=fs)
    # for frame_num in frames.keys():
    #     vid, full_color, fs = ofirs.load_video(frames[frame_num]["stab"])
    #     image = cv2.imread(frames[frame_num]['path'])
    #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #     img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     img_bin = img_gray > 20
    #     img_bin = img_bin.astype(int)
    #     mask = cv2.merge([img_bin, img_bin, img_bin])
    #
    #     fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #     shape_out = (full_color.shape[2],full_color.shape[1])
    #     out_path = os.path.join(new_frames_path, frames[frame_num]['label'] + "_vid", frame_num + '_part.avi')
    #     frames[frame_num]['part'] = out_path
    #     out = cv2.VideoWriter(out_path, fourcc, float(fs), shape_out)
    #     for i in range(full_color.shape[0]):
    #         img = np.multiply(full_color[i], mask).astype(np.uint8)
    #         # Convert the frame to a proper image format
    #         # Write the frame to the video file
    #         out.write(img.astype('uint8'))
    #     out.release()
    #     pass
    # #    utils.save_histograms(path)
    #
    #f = open("/home/stavb/new_frames/frames.pkl", "wb")
    ## write the python object (dict) to pickle file
    #pickle.dump(frames, f)
    ## close file
    #f.close()

    # from here we can load the dictionary and use it to fo over al the images

   #
   #  file_to_read = open("/home/stavb/new_frames/frames.pkl", "rb")
   #  frames = pickle.load(file_to_read)
   #
   #  frames_list=[]
   #  for frame_num,frame in enumerate(frames):
   #
   #      if utils.too_close(frame,frames_list):
   #          print("frame num ", frame_num, " frame :", frame, " skipped")
   #          frames_list.append(frame)
   #          continue
   #      # if frame_num < 8 :
   #      #     print("frame num ",frame_num," frame :",frame," skipped")
   #      #     frames_list.append(frame)
   #      #     continue
   #      frames_list.append(frame)
   #      print("working on frame num ", frame_num, " frame :", frame)
   #      vid, full_color, fs = ofirs.load_video(frames[frame]["part"],CHANGE2RGB = True)
   #      ofirs.time_corp_analyze_shell(full_color, stride=0, dur=4, left=0, top=0, right=full_color.shape[2],
   #                                     bottom=full_color.shape[1], label=frames[frame]["label"],fs=fs, name = frame)
   # #
   # #classification
    np_folder_path = '/home/stavb/PycharmProjects/BFE/Data/np_files_temp'
    classify.run_classifier_tagged_data_gab(np_folder_path)

   # np_folder_path = '/home/stavb/PycharmProjects/BFE/Data/np_files3'
   # classify.run_classifier_tagged_data_with_gabor(np_folder_path)

    np_format = 'npy'
    data = []

    for root, dirs, files in os.walk(np_folder_path):
        for filename in files:
                if filename.lower().endswith(np_format):
                    data.append(np.load(os.path.join(root,filename),allow_pickle=True))

    data = np.array(data,dtype = object)

    path_good = os.path.join(path,'good.xlsx')
    path_poor = os.path.join(path, 'poor.xlsx')
    #classify.run_classifier_hist(path_good,path_poor)
    #classify.run_classifier_fft('good_fft.npy', 'bad_fft.npy')

    #np_dir_path = r'/home/stavb/PycharmProjects/BFE/Data/np_files'
    #classify.run_classifier_ofir_data(np_dir_path)




    image_path = r'/home/stavb/PycharmProjects/BFE/Data/frames'
    images=[]
    for root, dirs, files in os.walk(image_path):
        for filename in files:
            if filename.lower().endswith(image_format):
                images.append(os.path.join(root, filename))

    annotate_path = r'/home/stavb/Annotated'
    good, poor = utils.get_label_list(annotate_path)

    vid, vid_fc = bfe.read_video(vid_path = r'/home/stavb/ch1_video_01.mpg', debug = True)



    pass

if __name__ == '__main__':
    main()



