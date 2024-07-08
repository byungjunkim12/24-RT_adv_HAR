import numpy as np
import scipy.io as sio
import pandas as pd
import argparse
import random

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
    dSamp = inputJson["dSampFactor"]
    padLen = inputJson["padLen"]
    bidirectional = inputJson["bidirectional"]
    nHidden = inputJson["nHidden"]
    nLayer = inputJson["nLayer"]
    batchSize = inputJson["batchSize"]
    LR = inputJson["learningRate"]
    dataWin = inputJson["dataWin"]
    fromInit = inputJson["fromInit"]
    
    inputJsonFile.close()

    if dataType == "JAR":
        fs = 320
        nSubC = 30
        nRX = 3
        activities = ['sitstill', 'falldown', 'liedown', 'standstill', 'walk', 'turn', 'stand', 'sit']
        
    elif dataType == "TAR":
        fs = 1000
        nSubC = 30
        nRX = 3
        activities = ['bed', 'fall', 'run', 'sitdown', 'standup', 'walk']
        # activities = ['fall', 'pickup', 'run', 'sitdown', 'standup', 'walk']

    print('json:', inputJsonFileName, 'device:', device)
    print('nHidden:', nHidden, 'bidirectional:', bidirectional, 'LR:', LR, 'fromInit:', fromInit, '# of classes:', len(activities))
    
    # load windowed data from csv file
    if dataType == "JAR":
        dataPath = dataPath + "HAR_" + dataType + "/dSamp_" + str(dSamp) + "/"         
    elif dataType == "TAR":
        if dataWin:
            dataPath = dataPath + "HAR_" + dataType + "/win_pad_" + str(padLen) + "/"         
        else:
            dataPath = dataPath + "HAR_" + dataType + "/noWin_dSamp_" + str(dSamp) +\
                "_pad_" + str(padLen) + "/" 
    dataDict = {file:[] for file in activities}
    for actInd, activity in enumerate(activities):
        longestLen = 0
        dataDict[activity] = defaultdict(list)

        # csvFileName = "/project/iarpa/wifiHAR/HAR_survey/noWin_pad_" + str(padLen) + "_" + activity + ".csv"
        if dataWin:
            xxLoadNP = np.load(dataPath + "xx_1000_60_" + activity + ".npy", allow_pickle=True)
            print(xxLoadNP.shape)
        else:
            xxLoadNP = np.load(dataPath + "xx_" + activity + ".npy", allow_pickle=True)
        for dataInd, data in enumerate(xxLoadNP):
            if data.shape[0] > longestLen:
                longestLen = data.shape[0]
            # print(data.shape)
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
        nTrData = np.floor(len(dataset)*0.5).astype(int)
        nTsData = np.floor(len(dataset)).astype(int) - nTrData

        if dataType == "JAR":
            # trDataList = random.sample(range(len(dataset)), nTrData)
            tsDataList = [512, 1, 2, 514, 4, 6, 7, 8, 519, 520, 523, 524, 13, 525, 15, 527, 17, 18, 530, 20, 21, 22, 533, 534, 535, 538, 28, 541, 30, 31, 32, 33, 547, 36, 37, 38, 39, 40, 548, 42, 549, 551, 553, 554, 47, 48, 49, 556, 51, 52, 53, 559, 562, 56, 57, 563, 564, 60, 61, 62, 63, 567, 65, 570, 67, 68, 69, 575, 71, 72, 583, 584, 75, 585, 77, 588, 79, 80, 81, 82, 83, 589, 85, 86, 590, 88, 592, 594, 91, 92, 93, 94, 95, 96, 99, 100, 102, 107, 109, 111, 112, 113, 116, 118, 124, 126, 128, 130, 133, 135, 137, 138, 140, 141, 146, 147, 150, 153, 154, 155, 156, 161, 163, 166, 167, 172, 173, 177, 178, 181, 182, 186, 187, 188, 190, 198, 199, 202, 203, 207, 208, 211, 212, 213, 214, 218, 220, 221, 223, 225, 226, 229, 230, 234, 235, 238, 239, 240, 241, 247, 252, 254, 255, 256, 257, 258, 259, 261, 262, 267, 270, 273, 274, 276, 277, 278, 279, 281, 282, 285, 287, 289, 291, 292, 293, 296, 297, 299, 301, 302, 303, 304, 309, 571, 315, 317, 572, 321, 322, 331, 336, 337, 339, 341, 344, 345, 347, 348, 350, 351, 353, 355, 356, 358, 360, 363, 369, 371, 373, 374, 375, 376, 377, 378, 379, 381, 388, 392, 394, 395, 398, 403, 404, 406, 407, 409, 410, 411, 412, 414, 415, 591, 420, 421, 422, 423, 424, 426, 427, 431, 434, 435, 436, 437, 438, 439, 440, 441, 447, 449, 451, 453, 454, 455, 599, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 470, 471, 472, 473, 474, 475, 477, 478, 479, 481, 482, 485, 487, 488, 490, 491, 494, 496, 500, 501, 502, 503, 504, 505, 506]
            trDataList = list(set(range(len(dataset))) - set(tsDataList))

            # trDataList = [512, 1, 2, 514, 4, 6, 7, 8, 519, 520, 523, 524, 13, 525, 15, 527, 17, 18, 530, 20, 21, 22, 533, 534, 535, 538, 28, 541, 30, 31, 32, 33, 547, 36, 37, 38, 39, 40, 548, 42, 549, 551, 553, 554, 47, 48, 49, 556, 51, 52, 53, 559, 562, 56, 57, 563, 564, 60, 61, 62, 63, 567, 65, 570, 67, 68, 69, 575, 71, 72, 583, 584, 75, 585, 77, 588, 79, 80, 81, 82, 83, 589, 85, 86, 590, 88, 592, 594, 91, 92, 93, 94, 95, 96, 99, 100, 102, 107, 109, 111, 112, 113, 116, 118, 124, 126, 128, 130, 133, 135, 137, 138, 140, 141, 146, 147, 150, 153, 154, 155, 156, 161, 163, 166, 167, 172, 173, 177, 178, 181, 182, 186, 187, 188, 190, 198, 199, 202, 203, 207, 208, 211, 212, 213, 214, 218, 220, 221, 223, 225, 226, 229, 230, 234, 235, 238, 239, 240, 241, 247, 252, 254, 255, 256, 257, 258, 259, 261, 262, 267, 270, 273, 274, 276, 277, 278, 279, 281, 282, 285, 287, 289, 291, 292, 293, 296, 297, 299, 301, 302, 303, 304, 309, 571, 315, 317, 572, 321, 322, 331, 336, 337, 339, 341, 344, 345, 347, 348, 350, 351, 353, 355, 356, 358, 360, 363, 369, 371, 373, 374, 375, 376, 377, 378, 379, 381, 388, 392, 394, 395, 398, 403, 404, 406, 407, 409, 410, 411, 412, 414, 415, 591, 420, 421, 422, 423, 424, 426, 427, 431, 434, 435, 436, 437, 438, 439, 440, 441, 447, 449, 451, 453, 454, 455, 599, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 470, 471, 472, 473, 474, 475, 477, 478, 479, 481, 482, 485, 487, 488, 490, 491, 494, 496, 500, 501, 502, 503, 504, 505, 506]
            # tsDataList = list(set(range(len(dataset))) - set(trDataList))

        elif dataType == "TAR":
            if dataWin:
                trDataList = random.sample(range(len(dataset)), int(0.5*len(dataset)))
                tsDataList = list(set(range(len(dataset))) - set(trDataList))
            else:
                tsDataList = [0, 1, 3, 4, 7, 10, 11, 13, 14, 15, 21, 22, 24, 27, 28, 29, 30, 32, 39, 40, 41, 42, 43, 46, 49, 52, 53, 55, 59, 60, 61, 62, 63, 67, 68, 69, 70, 73, 76, 78]
                trDataList = list(set(range(len(dataset))) - set(tsDataList))

        trData.append(torch.utils.data.Subset(dataset, trDataList))
        tsData.append(torch.utils.data.Subset(dataset, tsDataList))
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

    # HARNetSavePath = './savedModels/LSTM/' + dataType + '_' + modelName +\
    #     '_H_' + str(nHidden) + '_dS_' + str(dSamp) + '.cpkt'
    if dataWin:
        HARNetSavePath = './savedModels/LSTM/' + dataType + "_" + modelName + "_win_H_" + str(nHidden) + ".cpkt"
    else:
        HARNetSavePath = './savedModels/LSTM/' + dataType + "_" + modelName + "_H_" + str(nHidden) + "_dS_" + str(dSamp) + ".cpkt"

    if dataWin:
        HARNet = LSTMNet_TAR(nClasses=len(activities),\
                        input_size=nSubC*nRX,\
                        bidirectional=bidirectional,\
                        hidden_size=nHidden,\
                        num_layers=1,\
                        seq_length=1,\
                        device=device)
    else:
        HARNet = VariableLSTMNet(nClasses=len(activities),\
                        input_size=nSubC*nRX,\
                        bidirectional=bidirectional,\
                        hidden_size=nHidden,\
                        num_layers=1,\
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
        accTrain = getAcc(trLoader, padLen, HARNet, HARNet, variableLen=True)

        runningLoss = 0.0
        for tsInput, tsLabel in tsLoader:
            opt.zero_grad()
            tsInput = tsInput[:, padLen:, :]            
            tsOutput = HARNet(tsInput)
            tsLabel = tsLabel.to(device)
            tsloss = LSTMLoss(tsOutput, tsLabel)
            runningLoss += tsloss.item()

        avgTsLoss = runningLoss / len(tsLoader)
        accTest = getAcc(tsLoader, padLen, HARNet, HARNet, variableLen=True)

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