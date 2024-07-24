import json
import csv
import argparse
import numpy as np

from scipy import signal
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
    dSampFactor = inputJson["dSampFactor"]
    padLen = inputJson["padLen"]
    inputJsonFile.close()

    fs = 1000 # 1 kHz
    # dSampFactor = 4
    nSubC = 30
    nRX = 3
    
    activities = ['bed', 'fall', 'run', 'sitdown', 'standup', 'walk']

    if dataType == "TAR":
        dataPath = dataPathBase + "HAR_TAR/"
    dataDict = {activity:[] for activity in activities}
    for activity in activities:
        dataDict[activity] = defaultdict(list)
    noWindDataPath = dataPath + "noWin_dSamp_" + str(dSampFactor) + "_pad_" + str(padLen) + "/"

    for activity in activities:
        if dataType == "TAR":
            fileNameList = glob(dataPath + "input_csv/input_" + activity + "*.csv")
            outputXXFileName = noWindDataPath + "xx_" + activity

            print(activity, len(fileNameList))

            # xx = np.empty([0, (winLen//dSampFactor + padSize), nSubC*nRX], float)
        dataList = []

        for fileName in fileNameList:
            data = np.array([[float(elm) for elm in v] for v in csv.reader(open(fileName, 'r'))])
            annotFileName = fileName.replace('input', 'annotation')
            annot = np.array([[str(elm) for elm in v] for v in csv.reader(open(annotFileName, 'r'))])

            indActs = np.where(annot[:, 0] == activity)[0]
            indActsdSamp = indActs[::dSampFactor]
            # indActsdSamp = signal.resample_poly(indActs, 1, dSampFactor)
            indPad = np.arange(indActs[0]-padLen*dSampFactor, indActs[0], dSampFactor)
            indActswithPad = np.concatenate((indPad, indActsdSamp))
            
            dataActs = data[indActswithPad, 1:1+nSubC*nRX]
            dataList.append(dataActs)

        np.save(outputXXFileName, np.array(dataList, dtype=object), allow_pickle=True)

if __name__ == "__main__":
    main()
