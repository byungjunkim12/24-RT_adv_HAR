import json
import csv
import argparse
import numpy as np
import os

from scipy import signal
import scipy.io as sio
from collections import defaultdict
from glob import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--inputJson", type=str, help='input JSON filename')

    args = parser.parse_args()
    inputJsonFileName = args.inputJson
    inputJsonFile = open("./inputJson/genData/" + inputJsonFileName + ".json", "r")
    inputJson = json.load(inputJsonFile)
    dataPathBase = inputJson["dataPathBase"]
    dataType = inputJson["dataType"]
    dSampRate = inputJson["dSampRate"]
    inputJsonFile.close()

    fs = 320 # 320 kHz
    nSubC = 30
    nRX = 3
    activities = ['sitstill', 'falldown', 'liedown', 'standstill', 'walk', 'turn', 'stand', 'sit']

    dataPath = dataPathBase + "HAR_" + str(dataType) +  "/mat"
    dataDict = {activity:[] for activity in activities}
    for activity in activities:
        dataDict[activity] = defaultdict(list)
    saveDataPath = dataPathBase + "HAR_" + str(dataType) + "/dSamp_" + str(dSampRate) + "/"

    # print(saveDataPath)
    for subPath in os.walk(dataPath):
        print('processing', subPath[0])
        for fileName in subPath[2]:
            if fileName[-4:] != ".mat":
                continue
            actIdx = int(fileName.split('_')[3][-2:])
            match actIdx:
                case 1:
                    activity = 'sitstill'
                case 2:
                    activity = 'falldown'
                case 3:
                    activity = 'liedown'
                case 4:
                    activity = 'standstill'
                case 5:
                    activity = 'falldown'
                case 6:
                    activity = 'walk'
                case 7:
                    activity = 'turn'
                case 8:
                    activity = 'walk'
                case 9:
                    activity = 'turn'
                case 10:
                    activity = 'stand'
                case 11:
                    activity = 'sit'
                case 12:
                    continue
            
            loadedMat = sio.loadmat(subPath[0] + '/' + fileName)['data']
            dataCSI = np.empty((0,nSubC*nRX))
            for timeSamp in loadedMat:
                # print(timeSamp[0][0][0][-1].shape)
                dataCSI = np.append(dataCSI, np.reshape(np.squeeze(np.absolute(timeSamp[0][0][0][-1])),\
                                                        (1, nSubC*nRX)), axis=0)
                
            dataCSI_dSamp = dataCSI[::dSampRate, :]
            if dataCSI_dSamp.shape[0] == 0:
                print('skipping', fileName, dataCSI.shape[0])
                continue
            # print(dataCSI.shape, dataCSI_dSamp.shape)

            dataDict[activity]['input'].append(dataCSI_dSamp)
            # print(activity, dataCSI.shape, type(dataCSI), type(dataDict[activity]['input']))
        
    for activity in activities:
        outputXXFileName = saveDataPath + "xx_" + activity
        np.save(outputXXFileName, np.array(dataDict[activity]['input'], dtype=object), allow_pickle=True)

if __name__ == "__main__":
    main()
