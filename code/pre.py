from skimage import data
from skimage import color
import os
from vidstab import VidStab
from skimage.filters import meijering, sato, frangi, hessian, threshold_otsu, gaussian
import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage import exposure
from skimage.measure import label, regionprops, regionprops_table
from skimage.segmentation import active_contour
from skimage.color import rgb2gray
def F_frangi(img_adapteq):
    #cv2.imshow('image', img_adapteq)
    image_f = 10000*frangi(np.array(img_adapteq), sigmas=range(1, 3, 1), scale_range=None, scale_step=None, alpha=0.5, beta=0.5, gamma=15, black_ridges=True, mode='reflect', cval=0)
    #image_m = meijering(np.array(img_adapteq))
    #image_s = sato(np.array(img_adapteq),black_ridges=True, mode='reflect')
    #cv2.imshow('image', image_f)
    #cv2.waitKey(0)
    #plt.imshow(image_f, cmap="gray")
    #plt.show()
    return image_f
def threshold(image):
    thresh = threshold_otsu(image)
    print(thresh)
    binary = np.array(image > thresh, dtype=bool)
    plt.imshow(binary, cmap="gray")
    plt.show()
    return binary

def load_frame(path):
     # load the frame from path and return the equalize green channel
     img = cv2.imread(path)[:, :, 1]
     img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
     return img_adapteq

def video_stab():

    # Using defaults
    stabilizer = VidStab()
    stabilizer.stabilize(input_path='cut2.mov', output_path='stable_video.avi')

    # Using a specific keypoint detector
    stabilizer = VidStab(kp_method='ORB')
    # stabilizer.stabilize(input_path='cut2.mp4', output_path='stable_video.avi')

    # Using a specific keypoint detector and customizing keypoint parameters
    #stabilizer = VidStab(kp_method='FAST', threshold=42, nonmaxSuppression=False)
    #stabilizer.stabilize(input_path='cut2.mov', output_path='stable_video.avi')

def colon_seg():
    #img = data.astronaut()
    img = cv2.imread('/Users/ofirbenyosef/hello/frames/MicrosoftTeams-image (5).png')
    img = rgb2gray(img)

    s = np.linspace(0, 2*np.pi, 400)
    r = 240 + 300*np.sin(s)
    c = 370 + 350*np.cos(s)
    init = np.array([r, c]).T
    f_img = sato(np.array(img),black_ridges=True, mode='reflect')
    # = gaussian(img, 0.03, preserve_range=False)
    
    snake = active_contour(img, init, alpha=0.015, beta=10, gamma=0.001)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray)
    ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
    ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
    ax.set_xticks([]), ax.set_yticks([])
    ax.axis([0, img.shape[1], img.shape[0], 0])

    plt.show()

def load_video(path):
    # load the video from path
    cap = cv2.VideoCapture('/Users/ofirbenyosef/Desktop/OneDrive - Technion/מסמכים/מסמכים/סמסטר 8/פרוייקט ב/stable_video.avi')
    while (cap.isOpened()):
        ret, frame = cap.read()
        b_frame = np.uint8(1000*threshold(F_frangi(exposure.equalize_adapthist(frame[:, :, 1], clip_limit=0.03))))
        cv2.imshow('frame', b_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

def find_vessels(image):
    label_img = label(image)
    regions = regionprops(label_img)

    orientations = np.empty(regions.__len__())
    i = 0
    for prop in regions:
        orientations[i] = prop.orientation
        i += 1
    print('...')
    hist, bin_edges = np.histogram(orientations, density=True)
    _ = plt.hist(orientations)  # arguments are passed to np.histogram
    plt.title("Histogram of orientation")
    plt.show()


def ploting():
    path = r'/Users/ofirbenyosef/Desktop/OneDrive - Technion/מסמכים/מסמכים/סמסטר 8/פרוייקט ב/temp/frame00038.png'
    # Using cv2.imread() method
    img = cv2.imread(path)
    # Displaying the image
    cv2.imshow('image', img)
    cv2.waitKey(0)

def segment_images(path):
    for filename in os.listdir(path):
            frame = load_frame(os.path.join(path, filename))
            binary_frame = threshold(F_frangi(frame))
            #find_vessels(binary_frame)

if __name__ == '__main__':
    path = r'/Users/ofirbenyosef/hello/frames'
    #frame = load_frame(path)
    #binary_frame = threshold(F_frangi(frame))
    #find_vessels(binary_frame)
    #load_video(path)
    #video_stab()
    #segment_images(path)
    colon_seg()
