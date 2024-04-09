import numpy as np
import argparse
import json
import csv
from collections import defaultdict
from glob import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--inputJson", type=str, help='input JSON filename')

    args = parser.parse_args()
    inputJsonFileName = args.inputJson
    inputJsonFile = open("./inputJson/" + inputJsonFileName + ".json", "r")
    inputJson = json.load(inputJsonFile)
    dataPathBase = inputJson["dataPathBase"]
    dataType = inputJson["dataType"]
    thres = inputJson["threshold"]
    padLen = inputJson["padLen"]
    inputJsonFile.close()

    if dataType == "survey":
        fs = 1000
        nSubC = 30
        nRX = 3
        winLen = 1000
        slideLen = 400
        dSampFactor = 2

        activities = ['fall', 'bed', 'run', 'sitdown', 'standup', 'walk']


    if dataType == "survey":
        dataPath = dataPathBase + "HAR_survey/"
        dataDict = {activity:[] for activity in activities}
        for activity in activities:
            dataDict[activity] = defaultdict(list)
        windDataPath = dataPath + "noWin_pad_" + str(padLen) + "/"

    elif dataType == "SHARP":
        dataPath = dataPathBase + "HAR_SHARP/"

    for activity in activities:
        if dataType == "survey":
            fileNameList = glob(dataPath + "/input_" + activity + "*.csv")
            outputXXFileName = windDataPath + "xx_" + str(winLen) + "_" + str(thres) + "_" + activity + ".csv"
            outputYYFileName = windDataPath + "yy_" + str(winLen) + "_" + str(thres) + "_" + activity + ".csv"
        print('Processing', activity+", # of files:", len(fileNameList))

        xx = np.empty([0, (winLen//dSampFactor + padLen), nSubC*nRX], float)
        for fileName in fileNameList:
            data = np.array([[float(elm) for elm in v] for v in csv.reader(open(fileName, 'r'))])
            # x2 = np.empty([0, (winLen//dSampFactor + padLen), nSubC*nRX], float)

            annotFileName = fileName.replace('input', 'annotation')
            annot = np.array([[str(elm) for elm in v] for v in csv.reader(open(annotFileName, 'r'))])

            # k = padLen * dSampFactor
            # smallestFirstJ = winLen
            for k in range(slideLen, len(data)-winLen, slideLen):
                ySamp = np.array(annot[k:k+winLen])
                nActIndices = np.where(ySamp != 'NoActivity')[0].size
                if nActIndices > winLen * thres / 100:
                    xPad = np.dstack(data[k-padLen*dSampFactor:\
                                          k+winLen:dSampFactor, 1:1+nSubC*nRX].T)
                    xx = np.concatenate((xx, xPad), axis=0)
        xx = xx.reshape(len(xx), -1)

        with open(outputXXFileName, "w+") as f:
            writer = csv.writer(f)
            writer.writerows(xx)

if __name__ == "__main__":
    main()