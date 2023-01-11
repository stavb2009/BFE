from socket import INADDR_BROADCAST
from turtle import shape
from skimage import data
from skimage import color
import os
from datetime import date

# from vidstab import VidStab
from skimage.filters import meijering, sato, frangi, hessian, threshold_otsu
from skimage.filters import gaussian, laplace, sobel, roberts, gabor, gabor_kernel
from skimage.restoration import denoise_tv_chambolle
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
from scipy.fft import fftshift
import time
from sklearn.decomposition import PCA
from numpy.fft import fft, ifft
import random
import time


def load_frame(path):
    # load the frame from path and return the equalize green channel
    # img = cv2.imread(path)[:, :, 1]
    img = cv2.imread(path)
    temp = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    return img_adapteq


def load_video(path='stable_video_2.avi',start = False,duration = False, show = False, CHANGE2RGB = False):
    """
    get path and load the video in path

    return np array [t,w,h,3], [t,w,h], fs
    start : starting frame. default 0
    duration: in seconds. default all
    """
    cap = cv2.VideoCapture(path)

    n = int(cap.get(5))
    w = int(cap.get(3))
    h = int(cap.get(4))
    print(n, w, h)
    vid = []
    vid_fc = []
    count = 0
    while (cap.isOpened()):
        # vid_capture.read() methods returns a tuple, first element is a bool
        # and the second is frame
        ret, frame = cap.read()
        if ret == True:
            count = count + 1
            if start:
                if count < start: continue
            if show : cv2.imshow('Frame',frame)
            vid.append(frame[:, :, 1])
            if CHANGE2RGB:
                vid_fc.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                vid_fc.append(frame)

            # 20 is in milliseconds, try to increase the value, say 50 and observe
            key = cv2.waitKey(10)
            if duration:
                if len(vid) >= duration*n: break
            if key == ord('q'):
                break
        else:
            break
    # Release the video capture object
    cv2.destroyAllWindows()
    cap.release()
    vid = np.array(vid)
    vid_fc = np.array(vid_fc)
    return vid, vid_fc, n


def stft_2d(video):
    """
    get np array [t,w,h]
    go over each pixel in the video and calculate the STFT of it

    """
    fs = 24
    for i in range(0, video.shape[1]):
        for j in range(video.shape[2]):
            x = video[:, i, j]
            f, t, Zxx = signal.stft(x, fs, nperseg=128, window='boxcar')
            plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=5, shading='gouraud')
            plt.title('STFT Magnitude')
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.colorbar()
            plt.show()
            print(i, j)


def crop_good(vid):
    # crop video ro contain area with good blood flow
    left = 40
    top = 110
    right = 110
    bottom = 390
    good = vid[:, top:bottom, left:right]
    play_vid(good)
    # stft_2d(good)
    print('good')
    shell(good, 'good')


def crop_bad(vid):
    # crop video ro contain area with bad blood flow
    left = 270
    top = 150
    right = 520
    bottom = 200
    bad = vid[:, top:bottom, left:right]
    play_vid(bad)
    # stft_2d(bad)
    print('bad')
    shell(bad, 'bad')


# (300,70) : (530,100)

def play_vid(vid):
    """
    get np array [t,w,h] / [t,w,h,3] and dispaly it

    """
    try:
        dim = vid.shape[3]
        crop_vid = vid[:, top:bottom, left:right, :]
        for fr in range(0, np.shape(vid)[0]):
            cv2.imshow('video', cv2.cvtColor(vid[fr, :, :, :], cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)
    except:
        for fr in range(0, np.shape(vid)[0]):
            cv2.imshow('video', vid[fr, :, :])
            cv2.waitKey(1)
    cv2.destroyAllWindows()


def crop_box(x, y, video, len=20):
    """
    get np array [t,w,h] / [t,w,h,3] and crop box for analsys in a [len X len]
    around given pixel (x,y)
    return np array [t,len,len]

    """
    left = int(y - np.ceil(len / 2))
    top = int(x - np.ceil(len / 2))
    right = int(y + np.ceil(len / 2))
    bottom = int(x + np.ceil(len / 2))
    try:
        dim = vid.shape[3]
        box = video[:, top:bottom, left:right, :]
    except:
        box = video[:, top:bottom, left:right]
    return box


def mean_box(video):
    """
    get np array [t,w,h]
    mean of the frame return 1 pixel for frame
    return np array [t,1]

    """
    mean_box = []
    for frame in range(video.shape[0]):
        mat = np.matrix(video[frame, :, :])
        mm = np.matrix.mean(mat)
        mean_box.append(mm)
    mean_box = np.array(mean_box)
    return mean_box


def shell(video, label, idx=0):
    fs = 24
    Stride = 10
    num_x = int(np.floor(video.shape[1] / Stride)) - 1
    num_y = int(np.floor(video.shape[2] / Stride)) - 1

    plt.figure()
    for i in range(0, num_x):  # still need to think on it
        for j in range(0, num_y):  # still need to think on it
            X = Stride + Stride * i
            Y = Stride + Stride * j
            box = crop_box(X, Y, video)
            x = mean_box(box)
            # f, t, Zxx = signal.stft(x, fs,nperseg=128)
            X = fftshift(fft(x))
            # sos = signal.butter(10, 10, 'low', fs=24, output='sos')
            # filtered = signal.sosfilt(sos, X)
            # iX = ifft(filtered)
            # f, t, Zxx = signal.stft(iX,fs=24, nperseg=72,noverlap=12,padded=False,window='hamming')

            #  plt.pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=5, shading='gouraud')
            # plt.title('STFT Magnitude')
            # plt.ylabel('Frequency [Hz]')
            # plt.xlabel('Time [sec]')
            # plt.colorbar()
            # X=fft(x)
            n = np.arange(len(X)) - len(X) / 2
            T = len(X) / 24
            freq = n / T
            temp = abs(X)
            temp[60] = 0
            temp2 = temp
            temp2[temp < temp.max() / 4] = 0

            plt.stem(freq, temp)
            plt.xlabel(label)
            plt.xlim(-0.1, 3)

            name = 'stft_out/' + str(label) + '_' + str(idx) + '.png'
            print(name)
            # plt.savefig(name)
            idx = idx + 1
            plt.pause(0.1)
            plt.close()
    return idx


def color_hist_1():
    file0 = 'frames/MicrosoftTeams-image (2).png'
    img_1 = cv2.imread(file0)
    color = ('b', 'g', 'r')
    plt.figure()
    for i, col in enumerate(color):
        histr_1 = cv2.calcHist([img_1], [i], None, [256], [0, 256])
        plt.plot(histr_1, color=col)
        plt.xlim([0, 256])
    plt.show()
    file1 = 'frames/MicrosoftTeams-image (10).png'
    img_2 = cv2.imread(file1)
    color = ('b', 'g', 'r')
    plt.figure()
    for i, col in enumerate(color):
        histr_2 = cv2.calcHist([img_2], [i], None, [256], [0, 256])
        plt.plot(histr_2, color=col)
        plt.xlim([0, 256])
    plt.show()
    for i, col in enumerate(color):
        histr_1 = cv2.calcHist([img_1], [i], None, [256], [0, 256])
        histr_2 = cv2.calcHist([img_2], [i], None, [256], [0, 256])
        plt.plot(np.divide(histr_2, histr_1), color=col)
        plt.xlim([0, 256])
    plt.show()


def color_hist(frame):
    """
    get np array [w,h,3]
    calculate the color (RGB) histogram
    return np array [256,3]

    """
    temp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    color_1 = ('h', 's', 'v')
    color = ('r', 'g', 'b')
    histr = []
    for i, col in enumerate(color):
        histr.append(cv2.calcHist([temp], [i], None, [256], [0, 256]))

    return histr


def hist_shall(video):
    Stride = 10
    num_x = int(np.floor(video.shape[1] / Stride)) - 1
    num_y = int(np.floor(video.shape[2] / Stride)) - 1
    idx = 0

    for i in range(0, num_x):
        for j in range(0, num_y):
            X = Stride + Stride * i
            Y = Stride + Stride * j

            hists = []
            chis = []

            box = crop_box(X, Y, video)
            for ii in range(0, video.shape[0]):
                frame = box[ii, :, :, :]
                hists.append(color_hist(frame))
            hists = np.array(hists)
            print('...')

            for ii in range(1, len(hists)):
                chis.append(chi_square_hist(hists[0, :, :, 0], hists[ii, :, :, 0]))
            print('i = ' + str(i) + ' j = ' + str(j))
            chis = np.array(chis).transpose()
            plot_chi(chis)

            # time.sleep(1)


def chi_square_hist(hist1, hist2):
    chi = []
    for channel in range(0, hist1.shape[0]):
        chi.insert(channel, 0)
        temp = int(0)
        for i in range(0, hist1.shape[1]):
            if hist1[channel, i] == 0.0 and hist2[channel, i] == 0.0:
                temp += 0
            else:
                temp += (np.square(hist1[channel, i] - hist2[channel, i]) / (hist1[channel, i] + hist2[channel, i]))
        chi[channel] = temp
        # print(chi)
    return chi


def plot_chi(chi):
    chi = np.array(chi)
    plt.figure()
    color = ('r', 'g', 'b')
    color_1 = ('h', 's', 'v')
    for i, col in enumerate(color):
        plt.plot(chi[0, i, :], color=col)
        plt.legend(color_1)
        # plt.xlim([0,256])
    plt.pause(1)
    plt.close()


def showing(imgs):
    plt.figure()
    for i in imgs:
        plt.imshow(i, cmap=plt.cm.gray)
        plt.pause(0.1)
        plt.close()


def crop_vid(vid, left, top, right, bottom):
    try:
        dim = vid.shape[3]
        crop_vid = vid[:, top:bottom, left:right, :]
    except:
        crop_vid = vid[:, top:bottom, left:right]
    # play_vid(crop_vid)
    return crop_vid


def time_crop(video, dur=5, start=0):
    fs = 24
    num_of_frames = fs * dur
    end = start + num_of_frames
    try:
        dim = video.shape[3]
        crop_vid = video[start:end, :, :, :]
    except:
        crop_vid = video[start:end, :, :]
    # play_vid(crop_vid)
    return crop_vid, end


def texture_ana(vid, label):
    tiles = []
    Stride = 20
    num_x = int(np.floor(vid.shape[1] / Stride)) - 1
    num_y = int(np.floor(vid.shape[2] / Stride)) - 1
    pi = np.pi
    # (0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45)
    # name = 'stft_out/' +str(label) + '_' + str(idx) + '.png'
    for i in range(0, num_x):
        for j in range(0, num_y):
            X = Stride + Stride * i
            Y = Stride + Stride * j
            img = crop_box(X, Y, vid, len=40)
            for t in range(img.shape[0]):
                for freq in (0.05, 0.15, 0.25, 0.3, 0.45):
                    print(freq)
                    for theta_ in (0, pi / 4, pi / 2, 3 * pi / 4, pi, 5 * pi / 4, 6 * pi / 4, 7 * pi, 4):
                        print(theta_)
                        #temp = np.resize(cv2.cvtColor(img[t], cv2.COLOR_RGB2GRAY), (256, 256))
                        temp = cv2.cvtColor(img[t], cv2.COLOR_RGB2GRAY)
                        filt_real, filt_imag = gabor(temp, frequency=freq, theta=theta_)
                        tiles.append(filt_real)
                        name = 'gabor/' + str(label) + '/' + '_' + str(t) + str(j) + str(i) + '_' + str(
                            theta_) + '_' + str(freq)
                        # cv2.imwrite(name, filt_real)
                        np.save(name, filt_real)

    return tiles


def show_filters(filters):
    # plot
    fig = plt.figure(figsize=(10, 8))
    i = 1
    for filter in filters:
        ax = fig.add_subplot(8, 5, i)
        ax.imshow(np.array(filter))
        i += 1
    print('Hiiii')


def good_hist_prep(video):
    Stride = 10
    num_x = int(np.floor(video.shape[1] / Stride)) - 1
    num_y = int(np.floor(video.shape[2] / Stride)) - 1

    idx = 0
    f_hists = []
    for ii in range(0, video.shape[0]):
        hists = []
        for i in range(0, num_y):
            for j in range(0, num_x):
                X = Stride + Stride * i
                Y = Stride + Stride * j
                box = crop_box(X, Y, video)
                hists.append(color_hist(box[ii]))
        hists = np.array(hists)
        f_hists.append(np.average(hists, axis=0).astype(np.int32))
    return np.array(f_hists)


def time_corp_shell(video, stride, dur, left, top, right, bottom, label='bad'):
    Start = 0
    fs = 24
    end = video.shape[0]
    rep = int(np.floor(end / (dur * fs)))
    idx = 0
    for i in range(rep):
        croped, Start = time_crop(video, dur=5, start=Start)
        Start = Start - stride
        idx = shell(crop_vid(croped, left, top, right, bottom), label, idx)
        print(idx)


def hist_shall2(video, good_hist):
    Stride = 10
    num_x = int(np.floor(video.shape[1] / Stride)) - 1
    num_y = int(np.floor(video.shape[2] / Stride)) - 1
    idx = 0

    for i in range(0, num_x):
        for j in range(0, num_y):
            X = Stride + Stride * i
            Y = Stride + Stride * j

            hists = []
            chis = []

            box = crop_box(X, Y, video)
            for ii in range(0, video.shape[0]):
                frame = box[ii, :, :, :]
                hists.append(color_hist(frame))
            hists = np.array(hists)
            print('...')

            for ii in range(0, len(hists)):
                chis.append(chi_square_hist(good_hist[ii], hists[ii, :, :, 0]))
            print('i = ' + str(i) + ' j = ' + str(j))
            tv_denoised = denoise_tv_chambolle(np.array(chis), weight=70)
            chis = np.array(chis).transpose()
            tv_denoised = np.array(tv_denoised).transpose()

            plot_chi(tv_denoised)


def save2csv(histograms):
    f_out = open('good_hists.csv', "w")
    num = 0
    c = 0
    for hist in histograms:
        # f_out.write(str(num)+',')
        num += 1
        c = 0
        for color in hist:
            f_out.write(str(num) + ',' + str(c) + ',')
            c += 1
            for i in range(len(color) - 1):
                # print(str(color[i]) + ',')
                f_out.write(str(color[i][0]) + ',')
            f_out.write(str(color[i + 1][0]) + '\n')
    f_out.close()


def vid2gray(vid):
    """
    convert color video [t,X,Y,3] to gray

    """
    g_vid = []
    for i in range(vid.shape[0]):
        g_vid.append(cv2.cvtColor(vid[i, :, :, :], cv2.COLOR_RGB2GRAY))
    g_vid = np.array(g_vid)
    return g_vid


def FFT_module(vid, fs):
    """
    get np array video[t,x,y,3] and fs
    convert it to gray, calculate the mean of the frame  [t,X,Y] -> [t,Z]
    calculate the FFT of the signal and filters frequenceis with small magnitude
    return the filtered signal [1,len(FFT)]
    """
    # g_vid = vid2gray(vid)
    x = mean_box(vid[:, :, :, 1])
    X = fftshift(fft(x))
    n = np.arange(len(X)) - len(X) / 2
    T = len(X) / fs
    freq = n / T
    temp = abs(X)
    temp[int(len(x) / 2)] = 0
    temp2 = temp
    temp2[temp < temp.max() / 4] = 0
    return temp2


def color_module(frame):
    """
    get np array video[t,x,y,C]
    calculate each frame histogram and stack the 3 chanales to 1 vector 256X3 -> 768X1
    return vector tX768X1

    """
    hists = []
    #for ii in range(0, vid.shape[0]):
    #ii = random.randint(0,vid.shape[0]-1)
    #frame = vid[ii, :, :, :]
    C = color_hist(frame)
    C1 = np.reshape(C, -1)  # still need to check if work

    hists = np.array(C1)
    return hists


def texture_module(frame,gab):
    """
    get np array video[t,x,y,3] and convert it to gary [t,X,Y]
    calculate for each frame the gabor filters and stack each filter -> size 256X256
    return vector [t,NumOfFilters,67840]

    """
    tiles = []
    pi = np.pi
    freqs = (0.05, 0.15, 0.25, 0.3)
    # (0,pi/4,pi/2,3*pi/4,pi,5*pi/4,6*pi/4,7*pi/4)
    ang = (0, pi / 4, pi / 2, pi, 5 * pi / 4, 7 * pi / 4)
    #idxs = choose_random_frames(vid.shape[0], 1)

    #for idx in [idxs]:
    for freq in freqs:
        for theta_ in ang:
            #print(theta_)
            temp = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            #temp = cv2.resize(temp,(64,64))
            filt_real, filt_imag = gabor(temp, frequency=freq, theta=theta_)
            #G = np.reshape(filt_real[2:-2,2:-2], -1)
            if not gab : filt_real = np.reshape(filt_real, -1)
            tiles.append(filt_real)
    tiles = np.array(tiles)
    return tiles


def choose_random_frames(vid_len=120, num_of_frames=10):
    #return np.random.randint(1, vid_len, num_of_frames)
    return 0

def analyze_shell(video, fs=24, gab = False):
    """
    get np array video[t,x,y,3] and fs
    go over the video and crop tiles out of it and send to texture_module,
    color_module,FFT_module
    return list of [FFT,color,texture] by tiles

    """
    Stride = 10
    num_x = int(np.floor(video.shape[1] / Stride)) - 1
    num_y = int(np.floor(video.shape[2] / Stride)) - 1
    FFT = []
    color = []
    texture = []

    #results = np.zeros((num_x*num_y,3),dtype=object)
    FFT_results = []
    HIST_results = []
    texture_results = []
    HIST_hsv_results = []
    count = 0
    #times = np.zeros((5,15))
    print('get ready for ',num_x*num_y)
    for i in range(0, num_x):  # still need to think on it
        for j in range(0, num_y):  # still need to think on it
            #print(i," out of ",num_x," and ",j," out of ",num_y)
            X = Stride + Stride * i
            Y = Stride + Stride * j
            #times[0, count] = time.time()
            box = crop_box(X, Y, video)
            boxg = cv2.cvtColor(box[0], cv2.COLOR_RGB2GRAY)
            if boxg.mean() < 105:
                continue
            texture_results.append(texture_module(box[0],gab))
            if gab: continue
            #times[1,count] = time.time()
            FFT_results.append(FFT_module(box, fs))
            #times[2,count] = time.time()
            HIST_results.append(color_module(box[0]))
            #times[3,count] = time.time()
            #times[4,count] = time.time()
            HIST_hsv_results.append(color_module(cv2.cvtColor(box[0],cv2.COLOR_RGB2HSV)))
            count = count + 1

            if count%25 == 0 :
                print(count)
                pass
        #if count > 500: break  # to save time
            #    plt.imshow(box[0])
            #    plt.show()
    results = np.zeros((4, ), dtype=object)
    results[3] = np.array(HIST_hsv_results)
    results[2] = np.array(texture_results)
    results[1] = np.array(HIST_results)
    results[0] = np.array(FFT_results)


    return results


def time_corp_analyze_shell(video, stride, dur, left, top, right, bottom, label='bad', fs=24, name = '',gab=False):
    """
    get np array video[t,x,y,3] and fs
    crop video in time by dur (in sec) and place by left,top ,right ,bottom (in pixel)
    send the croped video to analyze_shell
    return the result in list [FFT,color,texture] by tiles
    """
    Start = 0

    end = video.shape[0]
    rep = int(np.floor(end / (dur * fs)))
    #result = []
    for i in range(rep):
        print("outer loop ",i," out of ",rep)
        croped, Start = time_crop(video, dur=dur, start=Start)
        Start = Start - stride
        result = analyze_shell(crop_vid(croped, left, top, right, bottom), fs, gab)

        name= date.today().strftime("%d_%m_%Y") + '_ofirs_'+label+'_'+ name + '_'  + str(i) + '.npy'
        save_path = os.path.join('/home/stavb/PycharmProjects/BFE/Data/np_files_temp',name)
        np.save(save_path,result)
    return 1


if __name__ == '__main__':
    #path = r'/Users/ofirbenyosef/hello/frames/MicrosoftTeams-image (9).png'
    # frame = load_frame(path)
    vid, full_color, fs = load_video('/home/stavb/ofir/stable_video_2.avi')
    # stft_2d(vid)
    # crop_good(vid)
    # crop_bad(vid)
    left = 270
    top = 150
    right = 520
    bottom = 200
    # hist_shall(crop_vid(full_color))
    # good_t = texture_ana(crop_vid(full_color[30:60,:,:,:],left = 40,top = 110,right = 110,bottom = 390),'good')
    # bad_t = texture_ana(crop_vid(full_color[30:60,:,:,:],left = 270,top = 150 ,right = 520,bottom = 200),'bad')
    # showing(bad_t)
    # show_filters(good_t)
    # show_filters(bad_t)
    # good_hist = good_hist_prep(crop_vid(full_color,left,top ,right,bottom))
    # hist_shall2(crop_vid(full_color,left = 40,top = 110,right = 110,bottom = 390),good_hist)
    # save2csv(good_hist)
    # showing(good_t)
    # time_corp_shell(vid,stride=0,dur=5,left=270,top=150,right=520,bottom=200,label = 'bad')
    # time_corp_shell(vid,stride=0,dur=5,left = 40,top = 110,right = 110,bottom = 390,label = 'good')
    Good = time_corp_analyze_shell(full_color, stride=0, dur=5, left=40, top=110, right=110, bottom=390, label='good',
                                   fs=fs)
    Bad = time_corp_analyze_shell(full_color, stride=0, dur=5, left=270,top=150,right=520,bottom=200, label='bad',
                                   fs=fs)
    print('HHH')




