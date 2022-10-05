from socket import INADDR_BROADCAST
from skimage import data
from skimage import color
import os
from vidstab import VidStab
from skimage.filters import meijering, sato, frangi, hessian, threshold_otsu
from skimage.filters import gaussian, laplace, sobel, roberts
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import exposure
from skimage.measure import label, regionprops, regionprops_table
from skimage.segmentation import active_contour
from skimage.color import rgb2gray
from skimage import filters
from skimage.filters.thresholding import _cross_entropy
import PIL
from PIL import Image

from scipy import ndimage as ndi
import plotly
import plotly.express as px
import plotly.graph_objects as go
from skimage import (
    color, feature, filters, measure, morphology, segmentation, util
)
from scipy import signal
def load_frame(path):
     # load the frame from path and return the equalize green channel
     img = cv2.imread(path)[:, :, 1]
     img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
     return img_adapteq

def go_over_frame(frame_t):
    fs = 10
    cap = cv2.VideoCapture('/Users/ofirbenyosef/hello/change_2.mov')
    
    n = int(cap.get(7))
    w = int(cap.get(3))
    h = int(cap.get(4))
    vid = []
    i = 0
    while(cap.isOpened()):
        # vid_capture.read() methods returns a tuple, first element is a bool 
        # and the second is frame
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow('Frame',frame)
            vid.append(frame[:,:,1])
            #i = i + 1
            # 20 is in milliseconds, try to increase the value, say 50 and observe
            key = cv2.waitKey(1)
            
            if key == ord('q'):
               break
        else:
            break
    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()
    #fig, axs = plt.subplots(1, 2,  sharey=True)
    #axs[0].imshow(frame_t)
    #fig.show()

    vid = np.array(vid)
    
    for i in range(200,576):
        for j in range(200,720):
                x = vid[:,i,j]
                f, t, Zxx = signal.stft(x, fs, nperseg=100)
                plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=70, shading='gouraud')
                #axs[1].imshow(mash)
                plt.title('STFT Magnitude,pic ')
                plt.ylabel('Frequency [Hz]')
                plt.xlabel('Time [sec]')
                

if __name__ == '__main__':
    path = r'/Users/ofirbenyosef/hello/frames/MicrosoftTeams-image (8).png'
    frame = load_frame(path)
    plt.imshow(frame),plt.colorbar(),plt.show()
    go_over_frame(frame)

   
