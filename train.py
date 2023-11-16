import numpy as np
import scipy.io as sio
import pandas as pd

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
import argparse

from glob import glob
from collections import defaultdict

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
    
    inputJsonFile = open("./inputJson/" + inputJsonFileName + ".json", "r")
    inputJson = json.load(inputJsonFile)
    dataPath = inputJson["dataPathBase"]
    dataType = inputJson["dataType"]
    nHidden = inputJson["nHidden"]
    thres = inputJson["threshold"]
    batchSize = inputJson["batchSize"]
    LR = inputJson["learningRate"]
    fromInit = inputJson["fromInit"]
    
    inputJsonFile.close()

    if dataType == "survey":
        fs = 1000
        nSubC = 30
        nRX = 3
        
        winLen = 1000
        sideLen = 400

        # activities = ['bed', 'fall', 'pickup', 'run', 'sitdown', 'standup', 'walk']
        activities = ['fall', 'pickup', 'run', 'sitdown', 'standup', 'walk']

    print('json:', inputJsonFileName, 'device:', device)
    print('nHidden:', nHidden, 'thres:', thres, 'batchSize:', batchSize, 'LR:', LR, 'fromInit:', fromInit, '# of classes:', len(activities))
    
    # load windowed data from csv file
    dataDict = {file:[] for file in activities}
    for actInd, activity in enumerate(activities):
        dataDict[activity] = defaultdict(list)

        csvFileName = "/project/iarpa/wifiHAR/HAR_survey/window/xx_1000_" + str(thres) + "_" + activity + ".csv"
        xxLoadNP = np.array(pd.read_csv(csvFileName, header=None))
        xxLoadNPReshape = xxLoadNP.reshape((len(xxLoadNP), winLen//2, nSubC*nRX))
        dataDict[activity]['input'] = xxLoadNPReshape
        dataDict[activity]['label'] = actInd*np.ones((xxLoadNP.shape[0]), dtype=int)
        # print(activity+":", dataDict[activity]['input'].shape)

    
    # split data into train and test
    trData = list()
    tsData = list()
    print('# of tr, ts, and total data: ')
    for activity in activities:
        dataset = CSIDataset(dataDict[activity], device)
        nTrData = np.floor(len(dataset)*0.8).astype(int)
        nTsData = np.floor(len(dataset)*0.2).astype(int)
        # nTrData = 173 # thres 0.6: 173, 0.8: 244
        # nTsData = 43 # thres 0.6: 43, 0.8: 61

        trData.append(torch.utils.data.Subset(dataset, range(nTrData)))
        tsData.append(torch.utils.data.Subset(dataset, range(nTrData, nTrData+nTsData)))
        print(activity, nTrData, nTsData, len(dataset))

    trDataset = torch.utils.data.ConcatDataset(trData) # concatenating dataset lists
    tsDataset = torch.utils.data.ConcatDataset(tsData)

    trLoader = DataLoader(trDataset, batch_size=batchSize, shuffle=True)
    tsLoader = DataLoader(tsDataset, batch_size=batchSize, shuffle=True)
    print("# of tr data:", len(trDataset), "# of ts data:", len(tsDataset))

    LSTMLoss = torch.nn.CrossEntropyLoss()
    maxPatience = 40
    nEpoch = 500
    HARNetSavePath = './savedModels/LSTM_H_' + str(nHidden) + '_th_' + str(thres) + '_B_'\
        + str(batchSize) + '_C_' + str(len(activities)) + '.cpkt'

    HARNet = LSTMNet(nClasses=len(activities), input_size=nSubC*nRX,\
        hidden_size=nHidden, num_layers=1, seq_length=winLen//2, device=device)
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
        for trIdx, trData in enumerate(trLoader):
            opt.zero_grad()
            trInput = trData['input']
            trLabel = trData['label']
            
            trOutput = HARNet(trInput)
            # print(trOutput)
            trloss = LSTMLoss(trOutput, trLabel)
            trloss.backward()
            opt.step()
            runningLoss += trloss.item()
            
        avgTrLoss = runningLoss / len(trLoader)
        accTrain = getAcc(trLoader, HARNet)

        runningLoss = 0.0
        for tsIdx, tsData in enumerate(tsLoader):
            opt.zero_grad()
            tsInput = tsData['input']
            tsLabel = tsData['label']
            
            tsOutput = HARNet(tsInput)
            tsloss = LSTMLoss(tsOutput, tsLabel)
            runningLoss += tsloss.item()

        avgTsLoss = runningLoss / len(tsLoader)
        accTest = getAcc(tsLoader, HARNet)

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