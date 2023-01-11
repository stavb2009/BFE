import cv2
import os
import numpy as np
import BFEpreprocessing as bfe
import pandas as pd
import sklearn
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import json
import random
from sklearn.neighbors import KNeighborsClassifier
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import seaborn as sns
from sklearn.decomposition import PCA
import time

def cuda(x):
    return x.cuda() if torch.cuda.is_available() else x
    #return x.cuda(non_blocking=True) if torch.cuda.is_available() else x


# class Net(nn.Module):
#     # for histograms
#   def __init__(self,input_shape):
#     super(Net,self).__init__()
#     self.fc1 = nn.Linear(input_shape,64)
#     self.fc2 = nn.Linear(64,128)
#     self.fc3 = nn.Linear(128, 16)
#     self.fc4 = nn.Linear(16, 1)
#   def forward(self,x):
#     x = torch.relu(self.fc1(x))
#     x = torch.relu(self.fc2(x))
#     x = torch.relu(self.fc3(x))
#     x = torch.sigmoid(self.fc4(x))
#     return x

class CIFAR_CNN(nn.Module):
    """CNN for the SVHN Datset"""

    def __init__(self):
        """CNN Builder."""
        super(CIFAR_CNN, self).__init__()

        # Conv Layer block 1
        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=24, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.1),

            # Conv Layer block 3
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.25),
        )

        self.residual_layer = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),

        )

        self.fc_layer1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True)
        )
        self.fc_layer2 = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True)
        )
        self.fc_layer3 = nn.Sequential(
            nn.Linear(256, 2)
        )

    def forward(self, x):
        """Perform forward."""
        # conv layers
        x = self.conv_layer(x)
        for i in range(3):
            x = self.residual_layer(x) + x
        # # flatten
        x = x.view(x.size(0), -1)

        # # fc layer
        x = self.fc_layer1(x)
        x = self.fc_layer2(x) + x
        x = self.fc_layer3(x)

        return x

def useCIFAR_CNN(X_train,y_train,X_test,y_test,sizes = False, show = False, lr = 0.001):
    ''''''
    # time to train our model
    # hyper-parameters
    batch_size = 128
    learning_rate = 1e-4
    epochs = 20

    # loss criterion
    criterion = nn.CrossEntropyLoss()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # build our model and send it to the device
    model = CIFAR_CNN().to(device)  # no need for parameters as we alredy defined them in the class

    # optimizer - SGD, Adam, RMSProp...
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    test_acc = []
    train_acc = []

    sc = StandardScaler()
    #X_train = sc.fit_transform(X_train)
    #X_test = sc.fit_transform(X_test)

    X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = torch.from_numpy(X_test).type(torch.FloatTensor)

    # training loop
    for epoch in range(1, epochs + 1):
        model.train()  # put in training mode
        running_loss = 0.0
        epoch_time = time.time()

        for i in range(X_train.shape[0]//batch_size):
            # get the inputs
            inputs = X_train[batch_size*i:batch_size*(i+1)]
            labels = y_train[batch_size * i:batch_size * (i + 1)]


            # forward + backward + optimize
            outputs = model(inputs)  # forward pass
            loss = criterion(outputs, labels)  # calculate the loss
            # always the same 3 steps
            optimizer.zero_grad()  # zero the parameter gradients
            loss.backward()  # backpropagation
            optimizer.step()  # update parameters

            # print statistics
            running_loss += loss.data.item()

        # Normalizing the loss by the total number of train batches
        running_loss /= len(X_train)

        # Calculate training/test set accuracy of the existing model
        outputs_train = model(X_train.to(device))
        outputs_test = model(X_test.to(device))
        acc_train = (outputs_train.reshape(-1).detach().cpu().numpy().round() == y_test).mean()
        acc_test = (outputs_test.reshape(-1).detach().cpu().numpy().round() == y_test).mean()

        epoch_time = time.time() - epoch_time
        log += "Epoch Time: {:.2f} secs".format(epoch_time)
        print(log)

        if i % (1) == 0:
            train_acc.append(loss)
            test_acc.append(acc_test)
            # print("epoch {}\tloss : {}\t accuracy : {}".format(i, loss, acc))
            log = "Epoch: {} | Loss: {:.4f} | Training accuracy: {:.3f}% | Test accuracy: {:.3f}% | ".format(epoch,
                                                                                                             running_loss,
                                                                                                             train_acc,
                                                                                                             test_acc)

    if show:
        # plotting the loss
        plt.plot(torch.tensor(losses, device='cpu'))
        plt.title('Loss vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.show()

        # printing the accuracy
        plt.plot(torch.tensor(accur, device='cpu'))
        plt.title('Accuracy vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()




        # # save model
        # if epoch % 20 == 0:
        #     print('==> Saving model ...')
        #     state = {
        #         'net': model.state_dict(),
        #         'epoch': epoch,
        #     }
        #     if not os.path.isdir('checkpoints'):
        #         os.mkdir('checkpoints')
        #     torch.save(state, './checkpoints/svhn_cnn_ckpt.pth')

    print('==> Finished Training ...')
    return(acc_test)


class Net(nn.Module):
  def __init__(self,input_shape):
    super(Net,self).__init__()
    self.fc1 = nn.Linear(input_shape,64)
    self.fc2 = nn.Linear(64,128)
    self.fc3 = nn.Linear(128, 16)
    self.fc4 = nn.Linear(128, 16)
    self.fc5 = nn.Linear(16, 1)
    self.input_shape = input_shape

  def forward(self,x):

    x = torch.relu(self.fc1(x))
    x = torch.relu(self.fc2(x))
    x = torch.relu(self.fc3(x))
    #x = torch.relu(self.fc4(x))
    x = torch.sigmoid(self.fc5(x))
    return x

  def set_params_size(self,sizes):
    self.fc1 = nn.Linear(self.input_shape, sizes[0])
    self.fc2 = nn.Linear(sizes[0], sizes[1])
    self.fc3 = nn.Linear(sizes[1], sizes[2])
    self.fc4 = nn.Linear(sizes[2], sizes[3])
    self.fc5 = nn.Linear(sizes[3], 1)

def useNN2(X_train,y_train,X_test,y_test,sizes = False, show = False, lr = 0.001):
    ''''''
    learning_rate = lr
    epochs = 15000
    model = Net(input_shape=X_train.shape[1])
    if sizes : model.set_params_size(sizes)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.BCELoss()

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
    #y_test = torch.from_numpy(y_test).type(torch.FloatTensor)
    X_test = X_test.to(device)
    #y_test = y_test.to(device)
    model = model.to(device)




    # forward loop
    losses = []
    accur = []
    for i in range(epochs):

        # calculate output
        output = model(X_train)

        # calculate loss
        loss = loss_fn(output, y_train.reshape(-1, 1))

        # accuracy
        predicted = model(torch.tensor(X_test, dtype=torch.float32))
        acc = (predicted.reshape(-1).detach().cpu().numpy().round() == y_test).mean()

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % (50) == 0:
            losses.append(loss)
            accur.append(acc)
            #print("epoch {}\tloss : {}\t accuracy : {}".format(i, loss, acc))

    if show:
        # plotting the loss
        plt.plot(torch.tensor(losses, device = 'cpu'))
        plt.title('Loss vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('loss')
        plt.show()

        # printing the accuracy
        plt.plot(torch.tensor(accur, device = 'cpu'))
        plt.title('Accuracy vs Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.show()

    pass
    output = predicted.reshape(-1).detach().cpu().numpy().round()
    return accur[-1] , output



def useNN(X,y):
    '''
    not in use
    :param X:
    :param y:
    :return:
    '''
    n_input, n_hidden, n_out, batch_size, learning_rate = 256, 15, 1, 100, 0.01
    epochs = 4000
    model = nn.Sequential(nn.Linear(n_input, n_hidden),
                          nn.ReLU(),
                          nn.Linear(n_hidden, n_out),
                          nn.Sigmoid())
    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X = torch.from_numpy(X).type(torch.FloatTensor)
    y = torch.from_numpy(y).type(torch.FloatTensor)
    X=X.to(device)
    y=y.to(device)
    model = model.to(device)


    losses = []
    for epoch in range(epochs):
        pred_y = model(X)
        loss = loss_function(pred_y, y)
        losses.append(loss.item())

        model.zero_grad()
        loss.backward()

        optimizer.step()

    plt.plot(losses)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title("Learning rate %f" % (learning_rate))
    plt.show()
    pass


def UseKNN(X_train,y_train,X_test,y_test, n_neighbors = 12):
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
    neigh.fit(X_train, y_train)
    score = neigh.score(X_test,y_test)
    return score


def UseLinearSVC(X_train,y_train,X_test,y_test):
    clf = make_pipeline(StandardScaler(),LinearSVC(random_state=0, tol=1e-5))
    clf.fit(X_train, y_train)
    score = clf.score(X_test,y_test)
    output = clf.predict(X_test)
    return score,output

def UseSVC(X_train,y_train,X_test,y_test):
    clf = make_pipeline(StandardScaler(), sklearn.svm.SVC(gamma='auto'))
    clf.fit(X_train, y_train)
    score = clf.score(X_test,y_test)
    output = clf.predict(X_test)
    return score, output





def upload_hist(path):
    hist = pd.read_excel(path , header=None)
    return hist.to_numpy()

def chi_square_hist(hist1,hist2):
    chi = []
    hist2=np.stack(hist2)
    for channel in range(0,hist1.shape[0]):
        chi.insert(channel,0)
        temp = int(0)
        for i in range(0,hist1.shape[1]):
            if hist1[channel,i] == 0.0 and hist2[channel,i] == 0.0:
                temp += 0
            else:
                temp += (np.square(hist1[channel,i] - hist2[channel,i])/(hist1[channel,i] + hist2[channel,i]))
        chi[channel] = temp
    return chi

def run_classifier_hist(path_good,path_poor):
    test_ratio = 0.85
    num_of_tries = 50
    print(path_good)
    good = pd.read_excel(path_good)
    poor = pd.read_excel(path_poor)
    i=15
    good_hist_path=r'/home/stavb/PycharmProjects/BFE/Data/good_hist.xlsx'
    good_hist = upload_hist(good_hist_path)

    X=[]
    red=[]
    green=[]
    blue=[]
    all=[]
    chis = []
    y=[]
    for i in range(1,good.shape[1]):
        red.append(return_array(good.iloc[2, i]))
        green.append(return_array(good.iloc[3, i]))
        blue.append(return_array(good.iloc[4, i]))
        all_tmp = good.iloc[2:5, i].to_numpy()
        for j in range(all_tmp.shape[0]):
            all_tmp[j] = return_array(all_tmp[j])
        all.append(np.concatenate(all_tmp))
        chis.append(chi_square_hist(good_hist, all_tmp))
        y.append(0)

    for i in range(1,poor.shape[1]):
        red.append(return_array(poor.iloc[2, i]))
        green.append(return_array(poor.iloc[3, i]))
        blue.append(return_array(poor.iloc[4, i]))
        all_tmp = poor.iloc[2:5, i].to_numpy()
        for j in range(all_tmp.shape[0]):
            all_tmp[j] = return_array(all_tmp[j])
        all.append(np.concatenate(all_tmp))
        chis.append(chi_square_hist(good_hist, all_tmp))
        y.append(1)


    #data_sets = [red, green, blue, all, chis]
    data_sets = [all]
    data_sets_names = ['red', 'green', 'blue', 'all', 'chis']
    scores1= np.zeros((num_of_tries,len(data_sets)))
    scores2 = np.zeros((num_of_tries, len(data_sets)))
    scores3 = np.zeros((num_of_tries, len(data_sets)))
    #scores_knn = np.zeros((20,num_of_tries, len(data_sets)))
    for try_num in range(num_of_tries):
        print("try num: ",try_num)
        #for k in range(1,20):

        for num, X in enumerate(data_sets):
            indices = [*range(len(X))]
            random.shuffle(indices)
            X=np.array(X)
            y=np.array(y)
            test_size = int(np.floor(len(X)*test_ratio))
            X_train = X[indices[:test_size]]
            y_train = y[indices[:test_size]]
            X_test = X[indices[test_size:]]
            y_test = y[indices[test_size:]]
            scores1[try_num,num] = UseLinearSVC(X_train,y_train,X_test,y_test)
            scores2[try_num, num] = UseSVC(X_train, y_train, X_test, y_test)
            #scores_knn[k,try_num, num] = UseKNN(X_train, y_train, X_test, y_test,k)
            #scores3[try_num,num]=useNN2(X_train,y_train,X_test,y_test,show=True)

    scores1 = scores1.mean(axis=0)
    scores2 = scores2.mean(axis=0)
    #scores_knn = scores_knn.mean(axis=1)
    file = r'/home/stavb/PycharmProjects/BFE/np.npy'
    np.save(file,scores3)
    pass

def run_classifier_fft(path_good,path_poor):
    test_ratio = 0.88
    num_of_tries = 50

    good = np.load('good_fft.npy')
    poor = np.load('bad_fft.npy')

    X=[]
    y=[]
    for i in range(good.shape[0]):
        for j in range(good.shape[1]):
            X.append(good[i][j])
            y.append(0)

    for i in range(poor.shape[0]):
        for j in range(poor.shape[1]):
            X.append(poor[i][j])
            y.append(1)


    scores1= np.zeros((num_of_tries,3))
    for try_num in range(num_of_tries):
        print("try num: ",try_num)
        #for k in range(1,20):
        indices = [*range(len(X))]
        random.shuffle(indices)
        X=np.array(X)
        y=np.array(y)
        test_size = int(np.floor(len(X)*test_ratio))
        X_train = X[indices[:test_size]]
        y_train = y[indices[:test_size]]
        X_test = X[indices[test_size:]]
        y_test = y[indices[test_size:]]
        #scores1[try_num,0] = UseLinearSVC(X_train,y_train,X_test,y_test)
        #scores1[try_num, 1] = UseSVC(X_train, y_train, X_test, y_test)
        #scores_knn[k,try_num, num] = UseKNN(X_train, y_train, X_test, y_test,k)
        scores1[try_num,2]=useNN2(X_train,y_train,X_test,y_test, show=True)
        pass

    scores1 = scores1.mean(axis=0)
    #scores2 = scores2.mean(axis=0)
    #scores_knn = scores_knn.mean(axis=1)
    #file = r'/home/stavb/PycharmProjects/BFE/np.npy'
    #np.save(file,scores3)
    pass

def run_classifier_ofir_data(path):
    test_ratio = 0.88
    num_of_tries = 100

    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    files = [f for f in onlyfiles if "ofir" in f]

    X_all = []
    X_fft = []
    X_hist = []
    y = []
    for file in files:
        print(file)
        np_file = os.path.join(path, file)
        A = np.load(np_file, allow_pickle=True)
        for i in range(A.shape[0]):
            X_all.append(np.concatenate([A[i,0],A[i,1]]))
            X_fft.append(A[i,0])
            X_hist.append(A[i, 1])
            if "good" in file:
                y.append(0)
            else:
                y.append(1)

    X_all = np.array(X_all)
    X_fft = np.array(X_fft)
    X_hist = np.array(X_hist)
    y = np.array(y)


    scores1= np.zeros((num_of_tries,3))
    X_list = [X_all, X_fft, X_hist]
    for try_num in range(num_of_tries):
        print("try num: ", try_num)
        for X_idx,X in enumerate(X_list):

            #for k in range(1,20):
            indices = [*range(X.shape[0])]
            random.shuffle(indices)
            test_size = int(np.floor(len(X)*test_ratio))
            X_train = X[indices[:test_size]]
            y_train = y[indices[:test_size]]
            X_test = X[indices[test_size:]]
            y_test = y[indices[test_size:]]
            #scores1[try_num,X_idx] = UseLinearSVC(X_train,y_train,X_test,y_test)
            #scores1[try_num, 1] = UseSVC(X_train, y_train, X_test, y_test)
            #scores_knn[k,try_num, num] = UseKNN(X_train, y_train, X_test, y_test,k)
            scores1[try_num,X_idx] =useNN2(X_train,y_train,X_test,y_test, show=True)
            pass

    scores1 = scores1.mean(axis=0)
    #scores2 = scores2.mean(axis=0)
    #scores_knn = scores_knn.mean(axis=1)
    #file = r'/home/stavb/PycharmProjects/BFE/np.npy'
    #np.save(file,scores3)
    pass

def run_classifier_tagged_data(np_folder_path):
    test_ratio = 0.8
    num_of_tries = 10

    np_format = 'npy'
    X_all = []
    X_fft = []
    X_hist = []
    X_hist_hsv = []
    X_gabor = []
    y = []
    count = 0
    count_save = 0
    for root, dirs, files in os.walk(np_folder_path):
        for filename in files:
            if filename.lower().endswith(np_format):
                print(filename)
                data = np.load(os.path.join(root, filename), allow_pickle=True)
                for i in range(data[0].shape[0]):
                    #X_fft.append(data[0][i])
                    #X_fft = data[0][i]
                    # hist = data[1][i]
                    # hist = hist.reshape(3, -1).T
                    # hist = hist[1:, :]
                    # hist = hist.reshape(-1, 1)
                    # X_hist.append(np.concatenate(hist, axis=0))
                    # #X_hist = np.concatenate(hist, axis=0)
                    #
                    # hist = data[3][i]
                    # hist = hist.reshape(3, -1).T
                    # hist = hist[1:, :]
                    # hist = hist.reshape(-1, 1)
                    # X_hist_hsv.append(np.concatenate(hist, axis=0))
                    X_gabor_item = np.concatenate(data[2][i].reshape(-1, 1), axis=0)
                    X_gabor.append(X_gabor_item)
                    #X_all.append(np.concatenate([X_fft, X_hist, X_gabor]))
                    #X_all.append(np.concatenate([X_fft, X_hist, X_gabor]))
                    if "Good" in filename:
                        y.append(0)
                    else:
                        y.append(1)
                count = count +1
                print("len is :",len(X_all))



    #
    X_fft  = np.array(X_fft)
    X_hist = np.array(X_hist)
    X_gabor = np.array(X_gabor)
    X_hist_hsv = np.array(X_hist_hsv)
    y = np.array(y)

    #np.save('/home/stavb/PycharmProjects/BFE/Data//np_files/X_all_20.npy', X_fft)
    #np.save('/home/stavb/PycharmProjects/BFE/Data//np_files/X_hist_20.npy', X_hist)
    #np.save('/home/stavb/PycharmProjects/BFE/Data//np_files/X_hist_hsv_20.npy', X_hist_hsv)
    np.save('/home/stavb/PycharmProjects/BFE/Data//np_files/X_gabor_gab.npy', X_gabor)
    np.save('/home/stavb/PycharmProjects/BFE/Data/np_files/y_gab.npy', y)


    #X_fft = np.load('/home/stavb/PycharmProjects/BFE/Data//np_files/X_all_20.npy',allow_pickle=True)
    #X_hist = np.load('/home/stavb/PycharmProjects/BFE/Data//np_files/X_hist_20.npy', allow_pickle=True)
    # X_hist_hsv = np.load('/home/stavb/PycharmProjects/BFE/Data//np_files/X_hist_hsv_20.npy', allow_pickle=True)
    # X_gabor = np.load('/home/stavb/PycharmProjects/BFE/Data//np_files/X_gabor_20_7.npy', allow_pickle=True)
    # y = np.load('/home/stavb/PycharmProjects/BFE/Data/np_files/y_all_20.npy',allow_pickle=True)
    #y2 = np.load('/home/stavb/PycharmProjects/BFE/Data/np_files/y_all_64.npy',allow_pickle=True)
    #X_all = X_all[:,861:]
    #X_no_gabor = X_all[:,:861]

    pca_100 = PCA(n_components=100)
    pca_100.fit(X_gabor.T)
    pca_500 = PCA(n_components=500)
    pca_500.fit(X_gabor.T)
    pca_1000 = PCA(n_components=1000)
    pca_1000.fit(X_gabor.T)




    X_fft_hist = np.concatenate([X_fft,X_hist],axis = 1)
    X_list = [X_fft,X_hist,pca_100.components_.T]
    scores = np.zeros((len(X_list), num_of_tries,3))
    outputs = np.zeros((len(X_list), num_of_tries,3), dtype = object)
    GT = np.zeros(num_of_tries, dtype = object)

    for try_num in range(num_of_tries):
        indices = [*range(X_fft.shape[0])]
        random.shuffle(indices)
        test_size = int(np.floor(len(X_fft) * test_ratio))
        for X_idx,X in enumerate(X_list):
            print("try num: ", try_num, " X idx ", X_idx)

            X_train = X[indices[:test_size]]
            y_train = y[indices[:test_size]]
            X_test = X[indices[test_size:]]
            y_test = y[indices[test_size:]]
            GT[try_num] = y_test
            scores[X_idx,try_num,0] ,outputs[X_idx,try_num,0] = UseLinearSVC(X_train,y_train,X_test,y_test)
            scores[X_idx,try_num, 1],outputs[X_idx,try_num,1] = UseSVC(X_train, y_train, X_test, y_test)
            scores[X_idx,try_num, 2] , outputs[X_idx,try_num,2] = useNN2(X_train, y_train, X_test, y_test, show=False, sizes=False, lr = 0.005)
            #scores1[X_idx,try_num,2] =useNN2(X_train,y_train,X_test,y_test, show=True,sizes = False)
            pass

    X = []
    y = []
    GT = np.stack(GT)
    for try1 in range(num_of_tries):
        A = np.stack(outputs[:, try1, 1])
        for indx in range(A.shape[1]):
            X.append(A[:, indx])
            y.append(GT[try1, indx])

    X = np.array(X)
    y = np.array(y)
    # sizes_arr = [4096,2048,1024,512,256,64,16]
    #
    # lrs = [0.001,0.01]
    #
    # scores = {}
    # count = 0
    # for size1 in sizes_arr:
    #     for size2 in sizes_arr:
    #         for size3 in sizes_arr:
    #             for size4 in sizes_arr:
    #                 for lr in lrs:
    #                     sizes = [size1, size2, size3, size4]
    #                     print("count = ", count, " size ",sizes)
    #                     if size1 == size2 or size1 == size3 or size4 >= size1 or size2 == size3 or size4 >= size3: continue
    #                     indices = [*range(X_all.shape[0])]
    #                     random.shuffle(indices)
    #                     test_size = int(np.floor(len(X_all)*test_ratio))
    #                     X_train = X_all[indices[:test_size]]
    #                     y_train = y[indices[:test_size]]
    #                     X_test = X_all[indices[test_size:]]
    #                     y_test = y[indices[test_size:]]
    #                     sizes = [size1,size2,size3,size4]
    #                     scores[count] ={}
    #                     scores[count]["sizes"]=sizes
    #                     scores[count]["acc"] = useNN2(X_train,y_train,X_test,y_test, show=False,sizes = sizes, lr = lr)
    #                     scores[count]["lr"] = lr
    #                     count = count + 1
    #                     #print(scores[count]["acc"])
    #
    #                     pass


    scores_avg = scores1.mean(axis=1)
    #useNN2(X_train, y_train, X_test, y_test, show=True)
    #scores2 = scores2.mean(axis=0)
    #scores_knn = scores_knn.mean(axis=1)
    #file = r'/home/stavb/PycharmProjects/BFE/np.npy'
    #np.save(file,scores3)
    pass

def run_classifier_tagged_data_gab(np_folder_path):
    test_ratio = 0.8
    num_of_tries = 1
    #
    # np_format = 'npy'
    # X_gabor = []
    # y = []
    # count = 0
    # count_save = 0
    # for root, dirs, files in os.walk(np_folder_path):
    #     for filename in files:
    #         if filename.lower().endswith(np_format):
    #             print(filename)
    #             data = np.load(os.path.join(root, filename), allow_pickle=True)
    #             for i in range(data[2].shape[0]):
    #
    #
    #                 X_gabor.append(data[2][i])
    #                 if "Good" in filename:
    #                     y.append(0)
    #                 else:
    #                     y.append(1)
    #             count = count +1
    # X_gabor = np.array(X_gabor)
    # y = np.array(y)
    # np.save('/home/stavb/PycharmProjects/BFE/Data//np_files/X_gabor_gab.npy', X_gabor)
    # np.save('/home/stavb/PycharmProjects/BFE/Data/np_files/y_gab.npy', y)

    X_gabor = np.load('/home/stavb/PycharmProjects/BFE/Data//np_files/X_gabor_gab.npy', allow_pickle=True)
    y = np.load('/home/stavb/PycharmProjects/BFE/Data/np_files/y_gab.npy',allow_pickle=True)

    X_gabor = X_gabor.reshape([-1,X_gabor.shape[1],20,20])

    X_list = [X_gabor]
    scores = np.zeros(num_of_tries)
    GT = np.zeros(num_of_tries, dtype = object)

    for try_num in range(num_of_tries):
        indices = [*range(X_gabor.shape[0])]
        random.shuffle(indices)
        test_size = int(np.floor(len(X_gabor) * test_ratio))
        for X_idx,X in enumerate(X_list):
            print("try num: ", try_num, " X idx ", X_idx)

            X_train = X[indices[:test_size]]
            y_train = y[indices[:test_size]]
            X_test = X[indices[test_size:]]
            y_test = y[indices[test_size:]]
            GT[try_num] = y_test
            scores[try_num]  = useCIFAR_CNN(X_train,y_train,X_test,y_test)
            pass

    X = []
    y = []
    GT = np.stack(GT)
    for try1 in range(num_of_tries):
        A = np.stack(outputs[:, try1, 1])
        for indx in range(A.shape[1]):
            X.append(A[:, indx])
            y.append(GT[try1, indx])

    X = np.array(X)
    y = np.array(y)
    # sizes_arr = [4096,2048,1024,512,256,64,16]
    #
    # lrs = [0.001,0.01]
    #
    # scores = {}
    # count = 0
    # for size1 in sizes_arr:
    #     for size2 in sizes_arr:
    #         for size3 in sizes_arr:
    #             for size4 in sizes_arr:
    #                 for lr in lrs:
    #                     sizes = [size1, size2, size3, size4]
    #                     print("count = ", count, " size ",sizes)
    #                     if size1 == size2 or size1 == size3 or size4 >= size1 or size2 == size3 or size4 >= size3: continue
    #                     indices = [*range(X_all.shape[0])]
    #                     random.shuffle(indices)
    #                     test_size = int(np.floor(len(X_all)*test_ratio))
    #                     X_train = X_all[indices[:test_size]]
    #                     y_train = y[indices[:test_size]]
    #                     X_test = X_all[indices[test_size:]]
    #                     y_test = y[indices[test_size:]]
    #                     sizes = [size1,size2,size3,size4]
    #                     scores[count] ={}
    #                     scores[count]["sizes"]=sizes
    #                     scores[count]["acc"] = useNN2(X_train,y_train,X_test,y_test, show=False,sizes = sizes, lr = lr)
    #                     scores[count]["lr"] = lr
    #                     count = count + 1
    #                     #print(scores[count]["acc"])
    #
    #                     pass


    scores_avg = scores1.mean(axis=1)
    #useNN2(X_train, y_train, X_test, y_test, show=True)
    #scores2 = scores2.mean(axis=0)
    #scores_knn = scores_knn.mean(axis=1)
    #file = r'/home/stavb/PycharmProjects/BFE/np.npy'
    #np.save(file,scores3)
    pass


def run_classifier_tagged_data_with_gabor(np_folder_path):
    test_ratio = 0.8
    num_of_tries = 50

    np_format = 'npy'
    X_all = []
    X_fft = []
    X_hist = []
    X_gabor = []
    y = []
    count = 0
    count_save = 0
    sample_ratio= 0.25
    file1 = r'/home/stavb/PycharmProjects/BFE/Data/with_gabor/1_2_3/1.npy'
    file2 = r'/home/stavb/PycharmProjects/BFE/Data/with_gabor/1_2_3/2.npy'
    file3 = r'/home/stavb/PycharmProjects/BFE/Data/with_gabor/1_2_3/3.npy'
    data1 = np.load(file1, allow_pickle=True)
    indices = [*range(data1.shape[0])]
    random.shuffle(indices)
    test_size = int(np.floor(len(data1) * sample_ratio))
    data1 = data1[indices[:test_size]]
    data2 = np.load(file2, allow_pickle=True)
    #data3 = np.load(file3, allow_pickle=True)
    data = np.concatenate([data1,data2])
    data1=[]
    data2=[]
    #data3=[]
    for root, dirs, files in os.walk(np_folder_path):
        for i,filename in enumerate(files):
            if filename.lower().endswith(np_format):
                print(filename)
                data[i] = np.load(os.path.join(root, filename), allow_pickle=True)

                #X_hist.append(np.concatenate(hist, axis=0))
                #X_hist = np.concatenate(hist, axis=0)
                #X_gabor = np.concatenate(data[2, 0][i].reshape(-1, 1), axis=0)
                #X_all.append(np.concatenate([X_fft, X_hist, X_gabor]))
                #X_all.append(np.concatenate([X_fft, X_hist]))
                if "Good" in filename:
                    y.append(0)
                else:
                    y.append(1)
                count = count +1
                print("len is :",len(X_all))

                print(count)
                # if count >6:
                #     count = 0
                #     X_all = np.array(X_all)
                #     print("count_save=",count_save)
                #     count_save = count_save +1
                #     name = str(count_save) + ".npy"
                #     save_path = os.path.join('/home/stavb/PycharmProjects/BFE/Data/np_files2', name)
                #     np.save(save_path, X_all)

                    # X_all=[]


    X_all = np.array(X_all)
    #X_fft = np.array(X_fft)
    #X_hist = np.array(X_hist)
    #X_gabor = np.array(X_gabor)
    y = np.array(y)


    scores1= np.zeros((num_of_tries,2))
    X_list = [X_all]
    for try_num in range(num_of_tries):
        print("try num: ", try_num)
        for X_idx,X in enumerate(X_list):

            #for k in range(1,20):
            indices = [*range(X.shape[0])]
            random.shuffle(indices)
            test_size = int(np.floor(len(X)*test_ratio))
            X_train = X[indices[:test_size]]
            y_train = y[indices[:test_size]]
            X_test = X[indices[test_size:]]
            y_test = y[indices[test_size:]]
            #scores1[try_num,0] = UseLinearSVC(X_train,y_train,X_test,y_test)
            #scores1[try_num, 1] = UseSVC(X_train, y_train, X_test, y_test)
            #scores_knn[k,try_num, num] = UseKNN(X_train, y_train, X_test, y_test,k)
            scores1[try_num,0] =useNN2(X_train,y_train,X_test,y_test, show=True)
            pass

    scores1 = scores1.mean(axis=0)
    #scores2 = scores2.mean(axis=0)
    #scores_knn = scores_knn.mean(axis=1)
    #file = r'/home/stavb/PycharmProjects/BFE/np.npy'
    #np.save(file,scores3)
    pass


def return_array(str):
    str = str.replace('\n', '').replace("[", '').replace("]", '')
    arr = np.fromstring(str,dtype=int, sep='.')
    return arr







if __name__ == '__main__':

    pass