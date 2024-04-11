import numpy as np
import scipy.io as sio
import pandas as pd
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import linalg as LA

import csv
import os
import json
import math
import time
import pickle

from glob import glob
from collections import defaultdict
from scipy import signal
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from utilities import *
from utilitiesDL import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--inputJson", type=str, help='input JSON filename')
    parser.add_argument('-c', "--cuda", type=int, default=0, help='cuda device number')
    
    args = parser.parse_args()
    inputJsonFileName = args.inputJson
    cudaDeviceNumber = args.cuda

    if cudaDeviceNumber >= 0:
        device = torch.device("cuda:" + str(cudaDeviceNumber))
    else:
        device = torch.device("cpu")
    
    inputJsonFile = open("./inputJson/LSTMJson/" + inputJsonFileName + ".json", "r")
    inputJson = json.load(inputJsonFile)
    dataPath = inputJson["dataPathBase"]
    dataType = inputJson["dataType"]
    nHidden = inputJson["nHidden"]
    nLayer = inputJson["nLayer"]
    batchSize = inputJson["batchSize"]
    LR = inputJson["learningRate"]
    bidirectional = inputJson["bidirectional"]
    padLen = inputJson["padLen"]
    fromInit = inputJson["fromInit"]
    
    inputJsonFile.close()

    if dataType == "survey":
        fs = 1000
        nSubC = 30
        nRX = 3
        dSampFactor = 2
        
        activities = ['bed', 'fall', 'run', 'sitdown', 'standup', 'walk']
        # activities = ['fall', 'pickup', 'run', 'sitdown', 'standup', 'walk']

    print('json:', inputJsonFileName, 'device:', device)
    print('nHidden:', nHidden, 'bidirectional:', bidirectional, 'LR:', LR, 'fromInit:', fromInit, '# of classes:', len(activities))
    
    # load windowed data from csv file
    dataPath = dataPath + "HAR_" + dataType + "/noWin_pad_" + str(padLen) + "/" 
    dataDict = {file:[] for file in activities}
    for actInd, activity in enumerate(activities):
        longestLen = 0
        dataDict[activity] = defaultdict(list)

        # csvFileName = "/project/iarpa/wifiHAR/HAR_survey/noWin_pad_" + str(padLen) + "_" + activity + ".csv"
        xxLoadNP = np.load(dataPath + "xx_" + activity + ".npy", allow_pickle=True)
        for dataInd, data in enumerate(xxLoadNP):
            if data.shape[0] > longestLen:
                longestLen = data.shape[0]

            dataDict[activity]['input'].append(data)
            dataDict[activity]['label'] = actInd*torch.ones((xxLoadNP.shape[0]), dtype=int, device=device)
        # print(activity+":", dataDict[activity]['input'].shape)

    # split data into train and test
    trData = list()
    tsData = list()
    print('# of tr, ts, and total data: ')
    for activity in activities:
        dataset = CSIDataset(dataDict[activity],\
                            device,\
                            normalize=True,\
                            nSubC=nSubC,\
                            nRX=nRX,\
                            padLen=padLen)
        nTrData = np.floor(len(dataset)*0.8).astype(int)
        nTsData = np.floor(len(dataset)*0.2).astype(int)

        trData.append(torch.utils.data.Subset(dataset, range(nTrData)))
        tsData.append(torch.utils.data.Subset(dataset, range(nTrData, nTrData+nTsData)))
        print(activity, nTrData, nTsData, len(dataset))

    trDataset = torch.utils.data.ConcatDataset(trData) # concatenating dataset lists
    tsDataset = torch.utils.data.ConcatDataset(tsData)

    trLoader = DataLoader(trDataset, batch_size=batchSize, shuffle=True, collate_fn=collate_fn)
    tsLoader = DataLoader(tsDataset, batch_size=batchSize, shuffle=True, collate_fn=collate_fn)
    print("# of tr data:", len(trDataset), "# of ts data:", len(tsDataset))

    LSTMLoss = torch.nn.CrossEntropyLoss()
    maxPatience = 40
    nEpoch = 500
    if bidirectional:
        modelName = "BiLSTM"
    else:
        modelName = "LSTM"

    HARNetSavePath = './savedModels/LSTM/' + modelName + '_H_' + str(nHidden) + '.cpkt'
    HARNet = VariableLSTMNet(nClasses=len(activities),\
                    input_size=nSubC*nRX,\
                    bidirectional=bidirectional,\
                    hidden_size=nHidden,\
                    num_layers=1,\
                    longestLen=longestLen,\
                    device=device)
    # HARNet = LSTMNet(nClasses=len(activities), bidirectional=bidirectional, input_size=nSubC*nRX,\
    #     hidden_size=nHidden, num_layers=nLayer, seq_length=winLen//2, device=device)

    HARNet.to(device)
    if fromInit:
        HARNet.apply(init_weights)
    else:
        HARNet.load_state_dict(torch.load(HARNetSavePath))
    opt = torch.optim.Adam(HARNet.parameters(), lr=LR)

    bestAcc = 0.0
    patience = 0
    fastConvg = True

    torch.set_num_threads(1)
    print('save model path:', HARNetSavePath)
    for epoch in range(nEpoch):
        runningLoss = 0.0
        HARNet.train()      
        for trInput, trLabel in trLoader:
            opt.zero_grad()
            trInput = trInput[:, padLen:, :]
            trOutput = HARNet(trInput)
            trLabel = trLabel.to(device)
            trloss = LSTMLoss(trOutput, trLabel)
            trloss.backward()
            opt.step()
            runningLoss += trloss.item()
            
        avgTrLoss = runningLoss / len(trLoader)
        accTrain = getAcc(trLoader, padLen, HARNet, variableLen=True)

        runningLoss = 0.0
        for tsInput, tsLabel in tsLoader:
            opt.zero_grad()
            tsInput = tsInput[:, padLen:, :]            
            tsOutput = HARNet(tsInput)
            tsLabel = tsLabel.to(device)
            tsloss = LSTMLoss(tsOutput, tsLabel)
            runningLoss += tsloss.item()

        avgTsLoss = runningLoss / len(tsLoader)
        accTest = getAcc(tsLoader, padLen, HARNet, variableLen=True)

        print('Epoch: %d, trLoss: %.3f, trAcc: %.3f, tsLoss: %.3f, tsAcc: %.3f'\
            % (epoch, avgTrLoss, accTrain, avgTsLoss, accTest))
        if bestAcc < accTest:
            bestAcc = accTest
            print('saving model')
            torch.save(HARNet.state_dict(), HARNetSavePath)  # saving model with best test accuracy
            patience = 0

        # early stopping if model converges twice
        patience += 1
        if patience > maxPatience:
            if fastConvg:
                LR = LR/10
                opt = torch.optim.Adam(HARNet.parameters(), lr=LR)
                patience = 0
                fastConvg = False
                print('fast convergence ends')
            else:
                break

    print('Training finished')


if __name__ == "__main__":
    main()