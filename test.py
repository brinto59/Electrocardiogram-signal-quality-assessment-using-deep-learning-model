import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import torchvision
import scipy.io as sio
from torch.utils.data import DataLoader,Dataset,TensorDataset
from torch.autograd import Variable
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.optim as optim
from keras.utils import to_categorical
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
import scipy.signal as signal
from sklearn.model_selection import StratifiedKFold
from torchsummary import summary


def zscore(data):
    data_mean=np.mean(data)
    data_std=np.std(data, axis=0)
    if data_std!=0:
        data=(data-data_mean)/data_std
    else:
        data=data-data_mean
    return data


class Layer(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(Layer, self).__init__()
        self.basic_module = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(out_ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(out_ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        )

    def forward(self, x):
        x = self.basic_module(x)
        # x = torchvision.ops.stochastic_depth(x, p=self.p, mode="row")
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batchnorm = nn.BatchNorm1d(out_ch, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x


class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_ch, out_ch, num_classes):
        super(AuxiliaryClassifier, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.conv1 = ConvBlock(in_ch, out_ch)
        # self.fc1 = nn.Linear(out_ch * 4, 256, bias=True)
        self.fc = nn.Linear(out_ch, num_classes, bias=True)
        self.activation = nn.ReLU()
        # self.dropout = nn.Dropout(0.7)
    def forward(self, x):
        out = self.pool(x)
        out = self.conv1(out)
        # print(out.size())
        out = out.view(out.size(0), out.size(1) * out.size(2))
        # print(out.size())
        # out = self.fc1(out)
        out = self.activation(out)
        # out = self.dropout(out)
        out = self.fc(out)

        return out


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.lrelu = nn.LeakyReLU()
        self.block0 = Layer(1, 64)
        self.block1 = Layer(64, 64)
        self.block2 = Layer(64, 128)
        self.block3 = Layer(128, 128)
        self.block4 = Layer(128, 256)
        self.block5 = Layer(256, 256)

        self.ai1 = AuxiliaryClassifier(64, 128, 2)
        self.ai2 = AuxiliaryClassifier(128, 256, 2)

        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.3, inplace=False)

        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc = nn.Linear(256, 2, bias=True)

    def forward(self, x):
        x0 = self.block0(x)
        x0 = self.maxpool1(x0)
        x0 = self.lrelu(x0)

        x1 = self.block1(x0)

        out1 = self.ai1(x1)

        x1 = self.maxpool1(x1)
        x1 = self.lrelu(x1)

        x2 = self.block2(x1)
        x2 = self.lrelu(x2)

        x3 = self.block3(x2)

        out2 = self.ai2(x3)
        x3 = self.lrelu(x3)

        x4 = self.block4(x3)
        x4 = self.lrelu(x4)

        x5 = self.block5(x4)
        x5 = self.lrelu(x5)

        out = self.avgpool(x5)
        out = self.dropout(out)
        out = out.view(out.size(0), out.size(1) * out.size(2))
        out = self.fc(out)

        return [out, out1, out2]




def test(model, testloader):
    model.eval()
    test_loss = 0.0
    test_correct = 0.0
    labelpredict = []
    label = []
    predict_proba = []
    data = []
    with torch.no_grad():
        for inputs1, labels1 in testloader:
            inputs1, labels1 = inputs1.cuda(), labels1.cuda()
            inputs1, labels1 = Variable(inputs1), Variable(labels1)
            output, _, _ = model(inputs1)

            data.append(inputs1.cpu().numpy())
            label.append(torch.argmax(labels1.cpu(), 1).numpy())
            predict_proba.extend(F.softmax(torch.FloatTensor(output.cpu()), dim=1)[:, 0].detach().numpy())
            labelpredict.append(torch.argmax(output.cpu(), 1))
    with open("predict_proba_by_model_fold_5.pkl", 'wb') as f:
        pickle.dump(predict_proba, f)
    return labelpredict, data


with open(f'testing_dataset.pkl', 'rb') as f:
    ecg_test, label_test = pickle.load(f)


ecgt = ecg_test
labelt = label_test
for FF1 in range(len(ecgt)):
    ecgt[FF1,:]=zscore(ecgt[FF1,:])

ecgt=torch.FloatTensor(ecgt)
ecgt=ecgt.unsqueeze(1)
labelt=to_categorical(labelt)
labelt=torch.FloatTensor(labelt)
deal_test_dataset = TensorDataset(ecgt,labelt)
testloader=DataLoader(dataset=deal_test_dataset,batch_size=32,shuffle=False,num_workers=0)

modelname= f"model_epoch15fold5.pkl"
model=torch.load(modelname)

model.eval()
labelpredict,testdata11 = test(model,testloader)
j1=[]
for j in labelpredict:
    j2=j.numpy()
    j1.extend(j2)

filename = f'label_predict_by_model_fold_0____.pkl'

with open(filename, 'wb') as f:
    pickle.dump(j1, f)