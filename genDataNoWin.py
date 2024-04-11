import json
import csv
import argparse
import numpy as np

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
    padLen = inputJson["padLen"]
    inputJsonFile.close()

    if dataType == 'survey':
        fs = 1000 # 1 kHz
        dSampFactor = 2
        nSubC = 30
        nRX = 3
        
        activities = ['bed', 'fall', 'run', 'sitdown', 'standup', 'walk']

    if dataType == "survey":
        dataPath = dataPathBase + "HAR_survey/"
    dataDict = {activity:[] for activity in activities}
    for activity in activities:
        dataDict[activity] = defaultdict(list)
    noWindDataPath = dataPath + "noWin_pad_" + str(padLen) + "/"

    for activity in activities:
        if dataType == "survey":
            fileNameList = glob(dataPath + "/input_" + activity + "*.csv")
            outputXXFileName = noWindDataPath + "xx_" + activity

            print(activity, len(fileNameList))

            # xx = np.empty([0, (winLen//dSampFactor + padSize), nSubC*nRX], float)
        dataList = []

        for fileName in fileNameList:
            data = np.array([[float(elm) for elm in v] for v in csv.reader(open(fileName, 'r'))])
            annotFileName = fileName.replace('input', 'annotation')
            annot = np.array([[str(elm) for elm in v] for v in csv.reader(open(annotFileName, 'r'))])

            indActs = np.where(annot[:, 0] == activity)[0]
            indPad = np.arange(indActs[0]-padLen*dSampFactor, indActs[0], dSampFactor)
            indActswithPad = np.concatenate((indPad, indActs))
            
            dataActs = data[indActswithPad, 1:1+nSubC*nRX]
            dataList.append(dataActs)

        np.save(outputXXFileName, np.array(dataList, dtype=object), allow_pickle=True)

if __name__ == "__main__":
    main()
