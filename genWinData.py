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
    padSize = inputJson["padSize"]
    inputJsonFile.close()

    if dataType == "survey":
        fs = 1000
        nSubC = 30
        nRX = 3
        winLen = 1000
        slideLen = 400
        dSampFactor = 2

        activities = ['bed', 'fall', 'pickup', 'run', 'sitdown', 'standup', 'walk']


    if dataType == "survey":
        dataPath = dataPathBase + "HAR_survey/"
        dataDict = {activity:[] for activity in activities}
        for activity in activities:
            dataDict[activity] = defaultdict(list)
        windDataPath = dataPath + "window_pad_9/"

    elif dataType == "SHARP":
        dataPath = dataPathBase + "HAR_SHARP/"

    for activity in activities:
        if dataType == "survey":
            fileNameList = glob(dataPath + "/input_" + activity + "*.csv")
            outputXXFileName = windDataPath + "xx_" + str(winLen) + "_" + str(thres) + "_" + activity + ".csv"
            outputYYFileName = windDataPath + "yy_" + str(winLen) + "_" + str(thres) + "_" + activity + ".csv"
        print('Processing', activity+", # of files:", len(fileNameList))

        xx = np.empty([0, (winLen//dSampFactor + padSize), nSubC*nRX], float)
        yy = np.empty([0, 8], float)
        for fileIndex, fileName in enumerate(fileNameList):
            data = np.array([[float(elm) for elm in v] for v in csv.reader(open(fileName, 'r'))])
            x2 = np.empty([0, (winLen//dSampFactor + padSize), nSubC*nRX], float)
            # y2 = np.empty([0, 8], float)
            # print(data.shape, end=' ')

            annotFileName = fileName.replace('input', 'annotation')
            annot = np.array([[str(elm) for elm in v] for v in csv.reader(open(annotFileName, 'r'))])

            k = padSize * dSampFactor
            smallestFirstJ = winLen
            while k <= (len(data) + 1 - winLen):
                x = np.dstack(data[(k-padSize*dSampFactor):k+winLen:dSampFactor, 1:1+nSubC*nRX].T)
                ySamp = np.array(annot[k:k+winLen])
                
                bed = 0
                fall = 0
                walk = 0
                pickup = 0
                run = 0
                sitdown = 0
                standup = 0
                noactivity = 0

                # firstJ = -1
                for j in range(winLen):
                    if ySamp[j] == "bed":
                        bed += 1
                    elif ySamp[j] == "fall":
                        fall += 1
                    elif ySamp[j] == "walk":
                        walk += 1
                    elif ySamp[j] == "pickup":
                        pickup += 1
                    elif ySamp[j] == "run":
                        run += 1
                    elif ySamp[j] == "sitdown":
                        sitdown += 1
                    elif ySamp[j] == "standup":
                        standup += 1
                    else:
                        noactivity += 1
                    
                    if ySamp[j] != "noactivity":
                        firstJ = j
                        if firstJ < smallestFirstJ:
                            smallestFirstJ = firstJ

                if bed > winLen * thres / 100:
                    yInd = 1
                elif fall > winLen * thres / 100:
                    yInd = 2
                elif walk > winLen * thres / 100:
                    yInd = 3
                elif pickup > winLen * thres / 100:
                    yInd = 4
                elif run > winLen * thres / 100:
                    yInd = 5
                elif sitdown > winLen * thres / 100:
                    yInd = 6
                elif standup > winLen * thres / 100:
                    yInd = 7
                else:
                    yInd = 0
                
                # print(x.shape, y.shape)
                if yInd != 0:
                    # xPad = np.dstack(data[k-padSize*dSampFactor:k+winLen-padSize:2, 1:1+nSubC*nRX].T)
                    x2 = np.concatenate((x2, x), axis=0)
                    # y2 = np.concatenate((y2, y), axis=0)
                k += slideLen

            xx = np.concatenate((xx, x2), axis=0)
            # print('activity', activity, 'smallestFirstJ:', smallestFirstJ)
            # yy = np.concatenate((yy, y2),axis=0)
            # print(y.shape)
        # print('Final smallestFirstJ:', smallestFirstJ)

        xx = xx.reshape(len(xx), -1)
        print(xx.shape)

        with open(outputXXFileName, "w+") as f:
            writer = csv.writer(f)
            writer.writerows(xx)
        # with open(outputYYFileName, "w") as f:
        #     writer = csv.writer(f)
        #     writer.writerows(yy)

if __name__ == "__main__":
    main()