# from socket import INADDR_BROADCAST
# from turtle import shape
# from skimage import data
# from skimage import color
# import os
# from vidstab import VidStab
# from skimage.filters import meijering, sato, frangi, hessian, threshold_otsu
# from skimage.filters import gaussian, laplace, sobel, roberts
# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
# from skimage import exposure
# from skimage.measure import label, regionprops, regionprops_table
# from skimage.segmentation import active_contour
# from skimage.color import rgb2gray
# from skimage import filters
# from skimage.filters.thresholding import _cross_entropy
# import PIL
# from PIL import Image
# from scipy import ndimage as ndi
# import plotly
# import plotly.express as px
# import plotly.graph_objects as go
# from skimage import (
#     color, feature, filters, measure, morphology, segmentation, util
# )
# from scipy import signal
# from scipy.fft import fftshift
# import generate_mask as gm
#
#
# def load_frame(path):
#     # load the frame from path and return the equalize green channel
#     # img = cv2.imread(path)[:, :, 1]
#     img = cv2.imread(path)
#     temp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     plt.imshow(cv2.cvtColor(temp, cv2.COLOR_RGB2HLS)), plt.show()
#     img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
#     return img_adapteq
#
#
# def open_video(frame_t):
#     # load video and save it as array to future work, can be change to get video for camara.
#     # save only the green channel <- change it !!!
#
#     cap = cv2.VideoCapture('/Users/ofirbenyosef/hello/stable_video.avi')
#
#     n = int(cap.get(5))
#     w = int(cap.get(3))
#     h = int(cap.get(4))
#     print(n, w, h)
#     vid = []
#     vid_fc = []
#     i = 0
#     while (cap.isOpened()):
#         # vid_capture.read() methods returns a tuple, first element is a bool
#         # and the second is frame
#         ret, frame = cap.read()
#         if ret == True:
#             cv2.imshow('Frame', frame)
#             vid.append(frame[:, :, 1])
#             vid_fc.append(frame)
#             # i = i + 1
#             # 20 is in milliseconds, try to increase the value, say 50 and observe
#             key = cv2.waitKey(10)
#
#             if key == ord('q'):
#                 break
#         else:
#             break
#     # Release the video capture object
#     cv2.destroyAllWindows()
#     cap.release()
#     vid = np.array(vid)
#     vid_fc = np.array(vid_fc)
#     return vid, vid_fc
#
#
# def stft_2d(video):
#     # go over each pixel in the video and calculate the STFT of it
#     fs = 24
#     for i in range(0, video.shape[1]):
#         for j in range(video.shape[2]):
#             x = video[:, i, j]
#             f, t, Zxx = signal.stft(x, fs, nperseg=128, window='boxcar')
#             plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=5, shading='gouraud')
#             plt.title('STFT Magnitude')
#             plt.ylabel('Frequency [Hz]')
#             plt.xlabel('Time [sec]')
#             plt.colorbar()
#             plt.show()
#             print(i, j)
#
#
# def crop_good(vid):
#     # crop video ro contain area with good blood flow
#     left = 40
#     top = 110
#     right = 110
#     bottom = 390
#     good = vid[:, top:bottom, left:right]
#     play_vid(good)
#     # stft_2d(good)
#     print('good')
#     shell(good, 'good')
#
#
# def crop_bad(vid):
#     # crop video ro contain area with bad blood flow
#     left = 270
#     top = 150
#     right = 520
#     bottom = 200
#     bad = vid[:, top:bottom, left:right]
#     play_vid(bad)
#     # stft_2d(bad)
#     print('bad')
#     shell(bad, 'bad')
#
#
# # (300,70) : (530,100)
#
# def play_vid(vid):
#     # play the video
#     for fr in range(0, np.shape(vid)[0]):
#         cv2.imshow('video', vid[fr, :, :,:])
#         cv2.waitKey(10)
#
#
# def crop_box(x, y, video):
#     # crop box for analsys in a [len X len]
#     # may change len
#     len = 20
#     left = int(y - np.ceil(len / 2))
#     top = int(x - np.ceil(len / 2))
#     right = int(y + np.ceil(len / 2))
#     bottom = int(x + np.ceil(len / 2))
#     box = video[:, top:bottom, left:right]
#     return box
#
#
# def mean_box(video):
#     # mean of the frame return 1 pixel for frame
#     mean_box = []
#     for frame in range(video.shape[0]):
#         mat = np.matrix(video[frame, :, :])
#         mm = np.matrix.mean(mat)
#         mean_box.append(mm)
#     mean_box = np.array(mean_box)
#     return mean_box
#
#
# def shell(video, label):
#     fs = 24
#     Stride = 10
#     num_x = int(np.floor(video.shape[1] / Stride)) - 1
#     num_y = int(np.floor(video.shape[2] / Stride)) - 1
#     idx = 0
#     for i in range(0, num_x):  # still need to think on it
#         for j in range(0, num_y):  # still need to think on it
#             X = Stride + Stride * i
#             Y = Stride + Stride * j
#             box = crop_box(X, Y, video)
#             x = mean_box(box)
#             f, t, Zxx = signal.stft(x, fs, nperseg=512, window='boxcar')
#             plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=5, shading='gouraud')
#             plt.title('STFT Magnitude')
#             plt.ylabel('Frequency [Hz]')
#             plt.xlabel('Time [sec]')
#             # plt.colorbar()
#             name = 'stft_out/' + str(label) + '_' + str(idx) + '.png'
#             print(name)
#             plt.save(name)
#             idx = idx + 1
#             plt.show()
#
#
# # def time_crop(video,size,fs):
# #     time = fs*size #num of frames in the time slot
# #     len = video.shape[0]
#
# def color_hist_1():
#     file0 = 'frames/MicrosoftTeams-image (2).png'
#     img_1 = cv2.imread(file0)
#     color = ('b', 'g', 'r')
#     plt.figure()
#     for i, col in enumerate(color):
#         histr_1 = cv2.calcHist([img_1], [i], None, [256], [0, 256])
#         plt.plot(histr_1, color=col)
#         plt.xlim([0, 256])
#     plt.show()
#     file1 = 'frames/MicrosoftTeams-image (10).png'
#     img_2 = cv2.imread(file1)
#     color = ('b', 'g', 'r')
#     plt.figure()
#     for i, col in enumerate(color):
#         histr_2 = cv2.calcHist([img_2], [i], None, [256], [0, 256])
#         plt.plot(histr_2, color=col)
#         plt.xlim([0, 256])
#     plt.show()
#     for i, col in enumerate(color):
#         histr_1 = cv2.calcHist([img_1], [i], None, [256], [0, 256])
#         histr_2 = cv2.calcHist([img_2], [i], None, [256], [0, 256])
#         plt.plot(np.divide(histr_2, histr_1), color=col)
#         plt.xlim([0, 256])
#     plt.show()
#
#
# def color_hist(frame):
#     # return color hist for HSV color space
#     # maybe need to change to numbers
#     temp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     img = cv2.cvtColor(temp, cv2.COLOR_RGB2HSV)
#     color_1 = ('h', 's', 'v')
#     color = ('r', 'g', 'b')
#     histr = []
#     # for i,col in enumerate(color_1):
#     #    histr.append(cv2.calcHist([img],[i],None,[256],[0,256]))
#     plt.figure()
#     for i, col in enumerate(color):
#         histr = cv2.calcHist([img], [i], None, [256], [0, 256])
#         plt.plot(histr, color=col)
#         plt.legend(color_1)
#         plt.xlim([0, 256])
#     plt.show()
#
#     return histr
#
#
# def hist_shall(video):
#     for i in range(0, video.shape[0]):
#         frame = video[i, :, :, :]
#         # plt.imshow(frame),plt.show()
#         color_hist(frame)
#
#
# if __name__ == '__main__':
#     path = r'/Users/ofirbenyosef/hello/frames/MicrosoftTeams-image (10).png'
#     #frame = load_frame(path)
#     #vid, full_color = open_video(frame)
#     # stft_2d(vid)
#     # crop_good(vid)
#     # crop_bad(vid)
#     #hist_shall(full_color)
#
#
