import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
import scipy.io as sio
from torch.utils.data import DataLoader,Dataset,TensorDataset
import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import StratifiedKFold
import copy


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
        self.fc = nn.Linear(out_ch, num_classes, bias=True)
        self.activation = nn.ReLU()
    def forward(self, x):
        out = self.pool(x)
        out = self.conv1(out)
        out = out.view(out.size(0), out.size(1) * out.size(2))
        out = self.activation(out)
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


def train(x, model, loss_train, acc_train):
    epoch = x
    j = x
    model.train()
    running_loss = 0.0
    train_correct = 0.0
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        output, out_1, out_2 = model(inputs)
        accuracy = (torch.argmax(output.cpu(), 1) == torch.argmax(labels.cpu(), 1)).numpy().sum()
        real_loss = criterion(output, torch.argmax(labels, dim=1))
        loss1 = criterion(out_1, torch.argmax(labels, dim=1))
        loss2 = criterion(out_2, torch.argmax(labels, dim=1))
        loss = real_loss + 0.3 * loss1 + 0.3 * loss2
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_correct += accuracy

    running_loss /= len(trainloader.dataset) / batch_size
    train_correct /= len(trainloader.dataset)
    acc_train[x] = train_correct
    los_train[x] = running_loss
    print('Epoch:%d train_loss:%.4f, train_correct:%.4f' % (epoch, running_loss, train_correct))
    return los_train, acc_train


def validation(j, model, los_test, acc_test):
    model.eval()
    test_loss = 0.0
    test_correct = 0.0
    with torch.no_grad():
        for inputs1, labels1 in testloader:
            inputs1, labels1 = inputs1.cuda(), labels1.cuda()
            # inputs1, labels1 = Variable(inputs1), Variable(labels1)
            output, _, _ = model(inputs1)
            accuracy1 = (torch.argmax(output.cpu(), 1) == torch.argmax(labels1.cpu(), 1)).numpy().sum()
            # accuracy1 = (output.cpu() == labels1.cpu()).numpy().sum()
            loss1 = criterion(output, torch.argmax(labels1, dim=1))
            test_loss += loss1.item()
            test_correct += accuracy1
    test_loss /= (len(testloader.dataset) / batch_size)
    test_correct /= (len(testloader.dataset))
    los_test[j] = test_loss
    acc_test[j] = test_correct
    print('test_loss:%.4f, test_correct%.4f' % (test_loss, test_correct))


with open('training_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

batch_size = 32
ecg = dataset[:, :-1]
label = dataset[:, -1]

print("Shape", ecg.shape)
print(pd.DataFrame(label).value_counts())

ecga, ecg_test, labela, label_test = train_test_split(ecg, label, test_size=0.2,random_state=42, shuffle=True)

with open(f'testing_dataset.pkl', 'wb') as f:
    testing_dataset = (ecg_test, label_test)
    pickle.dump(testing_dataset, f)

sfolder = StratifiedKFold(n_splits=7,random_state=1,shuffle=True)
for fold, (traindata, validationdata) in enumerate(sfolder.split(ecga,labela)):
    print(f'Fold: ', fold)
    ecgc = ecga[traindata]
    ecgt = ecga[validationdata]
    labelc = labela[traindata]
    labelt = labela[validationdata]

    for FF in range(len(ecgc)):
        ecgc[FF,:]=zscore(ecgc[FF,:])
    for FF1 in range(len(ecgt)):
        ecgt[FF1,:]=zscore(ecgt[FF1,:])

    ecgc=torch.FloatTensor(ecgc)
    ecgc=ecgc.unsqueeze(1)
    labelc=to_categorical(labelc)
    labelc=torch.FloatTensor(labelc)
    deal_dataset = TensorDataset(ecgc,labelc)
    trainloader=DataLoader(dataset=deal_dataset,batch_size=batch_size,shuffle=False,num_workers=0)

    ecgt=torch.FloatTensor(ecgt)
    ecgt=ecgt.unsqueeze(1)
    labelt=to_categorical(labelt)
    labelt=torch.FloatTensor(labelt)
    deal_test_dataset = TensorDataset(ecgt,labelt)
    testloader=DataLoader(dataset=deal_test_dataset,batch_size=batch_size,shuffle=False,num_workers=0)

    model = Model()
    model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    los_train=np.zeros(200)
    acc_train=np.zeros(200)
    los_valid=np.zeros(200)
    acc_valid=np.zeros(200)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.80, dampening=0.0, weight_decay=0.001, nesterov=False)

    best_accuracy = 0
    best_model = model
    matname = 'model_'+ 'epoch0' + 'fold0' + '.pkl'
    for x in range(200):
        los_train, acc_train=train(x,model,los_train, acc_train)
        validation(x,model,los_valid, acc_valid)
        if acc_valid[x]>=best_accuracy:
            matname = 'model_' + 'epoch' + str(x) + 'fold' + str(fold) + '.pkl'
            best_model = copy.deepcopy(model)
            best_accuracy = acc_valid[x]
    torch.save(best_model, matname)

    print("Max accuracy: ", np.max(acc_valid), "Index: ", np.argmax(acc_valid))
    filename_save='model_loss_accuracy_result'+'fold'+ str(fold) + '.mat'
    sio.savemat(filename_save, {'los_train': los_train,'acc_train': acc_train,'los_valid': los_valid,'acc_valid': acc_valid})

