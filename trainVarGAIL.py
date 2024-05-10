import numpy as np
import torch
import json
import random
import argparse

from torch.utils.data import DataLoader
from torch.nn import Module

from collections import defaultdict

# from GAIL.models.gail import GAIL
from GAIL.models.nets import PolicyNetwork, ValueNetwork, Discriminator
from GAIL.utils.funcs import get_flat_grads, get_flat_params, set_params, \
    conjugate_gradient, rescale_and_linesearch

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor

from utilities import *
from utilitiesDL import *



def main():
    # torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--inputJson", type=str, help='input JSON filename')
    parser.add_argument('-c', "--cuda", type=int, default=0, help='cuda device number')
    parser.add_argument("--load_model", action=argparse.BooleanOptionalAction,\
                        help='load model if model with the filename exists')

    args = parser.parse_args()
    inputJsonFileName = args.inputJson
    cudaID = args.cuda
    loadModel = args.load_model

    if cudaID >= 0:
        device = torch.device("cuda:" + str(cudaID))
    else:
        device = torch.device("cpu")

    LSTMModelDir = './savedModels/selected/'
    inputJsonFile = open("./inputJson/varGAIL/" + inputJsonFileName + ".json", "r")
    inputJson = json.load(inputJsonFile)
    inputJsonFile.close()

    LSTMTargetModelName = inputJson['LSTMTargetModelName']
    LSTMSurroModelName = inputJson['LSTMSurroModelName']
    noiseAmpRatio = inputJson['noiseAmpRatio']
    tsDataList = inputJson['tsDataList']
    trExpDataRatio = inputJson['trExpDataRatio']
    nHiddenGAIL = inputJson['nHiddenGAIL']
    GAILTrainConfig = inputJson['trainConfig']
    padLen = inputJson['padLen']
    inputLenTime = inputJson['inputLenTime']
    outputLenTime = inputJson['outputLenTime']
    dSampFactor = inputJson['dSampFactor']
    delayLen = inputJson['delayLen']
    trainAct = inputJson['trainAct']
    
    torch.set_num_threads(1)

    if cudaID >= 0:
        device = torch.device("cuda:"+str(cudaID))
    else:
        device = torch.device("cpu")

    dataType = LSTMTargetModelName.split('_')[0]
    if dataType == 'survey':
        fs = 1000 # 1 kHz
        nSubC = 30
        nRX = 3
        activities = ['bed', 'fall', 'run', 'sitdown', 'standup', 'walk']

    LSTMTargetType = LSTMTargetModelName.split('_')[2]
    bidirTarget = (LSTMTargetType == 'BiLSTM')
    LSTMSurroType = LSTMSurroModelName.split('_')[2]
    bidirSurro = (LSTMSurroType == 'BiLSTM')

    nHiddenTarget = int(LSTMTargetModelName.split('_')[4])
    nHiddenSurro = int(LSTMSurroModelName.split('_')[4])

    # Load the LSTM model
    HARTargetNet = VariableLSTMNet(nClasses=len(activities),\
                    input_size=nSubC*nRX,\
                    bidirectional=bidirTarget,\
                    hidden_size=nHiddenTarget,\
                    num_layers=1,\
                    device=device)
    HARTargetNet.load_state_dict(torch.load(LSTMModelDir + LSTMTargetModelName + '.cpkt', map_location=device))

    HARSurroNet = VariableLSTMNet(nClasses=len(activities),\
                    input_size=nSubC*nRX,\
                    bidirectional=bidirSurro,\
                    hidden_size=nHiddenSurro,\
                    num_layers=1,\
                    device=device)
    HARSurroNet.load_state_dict(torch.load(LSTMModelDir + LSTMSurroModelName + '.cpkt', map_location=device))

    # Load dataset labelled with FGM attack
    FGMdatasetDir = '/project/iarpa/wifiHAR/HAR_' + dataType + '/noWin_dSamp_' + str(dSampFactor) + '_pad_' + str(padLen) + '_FGM/'
    dataDict = {file:[] for file in activities}

    trExpDataset = list()
    trAgentDataset = list()
    tsDataset = list()
    for actInd, activity in enumerate(activities):
        if activity in trainAct:
            dataDict[activity] = defaultdict(list)
            dataInputActFileName = FGMdatasetDir + 'input_' + LSTMSurroModelName + '_' + activity + '.npy'
            dataFGMActFileName = FGMdatasetDir + 'FGM_' + LSTMSurroModelName + '_' + activity + '.npy'
            datasetObs = np.load(dataInputActFileName, allow_pickle=True)
            datasetFGM = np.load(dataFGMActFileName, allow_pickle=True)

            for (ob, FGM) in zip(datasetObs, datasetFGM):
                dataDict[activity]['obs'].append(ob)
                dataDict[activity]['FGM'].append(FGM)
            dataDict[activity]['label'] =\
                (actInd) * torch.ones_like(torch.empty(len(dataDict[activity]['obs']),\
                                                    device=device), dtype=int)
            
            trDataList = list(set(range(len(datasetObs))) - set(tsDataList))
            trExpDataList = random.sample(trDataList, int(trExpDataRatio*len(trDataList)))
            trAgentDataList = list(set(trDataList) - set(trExpDataList))

            datasetAct = FGMDataset(dataDict[activity], device, noiseAmpRatio=noiseAmpRatio, padLen=padLen)
            tsDataset.append(torch.utils.data.Subset(datasetAct, tsDataList))
            trExpDataset.append(torch.utils.data.Subset(datasetAct, trExpDataList))
            trAgentDataset.append(torch.utils.data.Subset(datasetAct, trAgentDataList))

            print('activity:', activity, 'trExpDataset:', len(trExpDataset[-1]),\
                'trAgentDataset:', len(trAgentDataset[-1]), 'tsDataset:', len(tsDataset[-1]))


    trExpLoader = DataLoader(torch.utils.data.ConcatDataset(trExpDataset),\
                        batch_size=1, shuffle=True, generator=torch.Generator(device=device))
    trAgentLoader = DataLoader(torch.utils.data.ConcatDataset(trAgentDataset),\
                        batch_size=1, shuffle=True, generator=torch.Generator(device=device))
    tsLoader = DataLoader(torch.utils.data.ConcatDataset(tsDataset),\
                        batch_size=1, shuffle=True, generator=torch.Generator(device=device))

    print('trExpLoader:', len(trExpLoader), 'trAgentLoader', len(trAgentLoader), 'tsLoader:', len(tsLoader))

    model = varGAIL(state_dim=nSubC*nRX,\
                action_dim=nSubC*nRX,\
                nHidden=nHiddenGAIL,\
                padLen=padLen,\
                inputLenTime=inputLenTime,\
                outputLenTime=outputLenTime,\
                delayLen=delayLen,\
                discrete=False,\
                device=device,\
                train_config=GAILTrainConfig)
    saveFileName = LSTMSurroModelName + '_' + inputJsonFileName
    if loadModel:
        model.pi.load_state_dict(torch.load('./savedModels/varGAIL/' + saveFileName + '_pi.cpkt'))
        model.v.load_state_dict(torch.load('./savedModels/varGAIL/' + saveFileName + '_v.cpkt'))
        model.d.load_state_dict(torch.load('./savedModels/varGAIL/' + saveFileName + '_d.cpkt'))
        print('model loaded!')

    model.train(HARTargetNet, trExpLoader, trAgentLoader, tsLoader, saveFileName)
    return 0

class varGAIL(Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        nHidden,
        padLen,
        inputLenTime,
        outputLenTime,
        delayLen,
        discrete,
        device,
        train_config=None
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.nHidden = nHidden
        self.discrete = discrete
        self.padLen = padLen
        self.inputLenTime = inputLenTime
        self.outputLenTime = outputLenTime
        self.delayLen = delayLen
        self.device = device
        self.train_config = train_config

        self.pi = PolicyNetwork(self.state_dim*self.inputLenTime,\
                                self.action_dim, self.nHidden,\
                                self.discrete, self.device)
        self.v = ValueNetwork(self.state_dim*self.inputLenTime,\
                              self.nHidden, self.device)
        self.d = Discriminator(self.state_dim*self.inputLenTime,\
                            self.action_dim, self.nHidden,\
                            self.discrete, self.device)

    def get_networks(self):
        return [self.pi, self.v]

    def act(self, state):
        self.pi.eval()
        state = FloatTensor(state)
        distb = self.pi(state)
        action = distb.sample()
        # action = distb.sample().detach().cpu().numpy()

        return action

    def train(self, HARTargetNet, trExpLoader, trAgentLoader, tsLoader, saveFileName):
        num_iters = self.train_config["num_iters"]
        lambda_ = self.train_config["lambda"]
        gae_gamma = self.train_config["gae_gamma"]
        gae_lambda = self.train_config["gae_lambda"]
        eps = self.train_config["epsilon"]
        max_kl = self.train_config["max_kl"]
        cg_damping = self.train_config["cg_damping"]
        opt_d_LR = self.train_config["opt_d_LR"]
        normalize_advantage = self.train_config["normalize_advantage"]

        opt_d = torch.optim.Adam(self.d.parameters(), lr=opt_d_LR)

        nDataTrExp = 0
        for _ in trExpLoader:
            nDataTrExp += 1
        nDataTrAgent = 0
        for _ in trAgentLoader:
            nDataTrAgent += 1
        nDataTs = 0
        for _ in tsLoader:
            nDataTs += 1
        print('nDataTrExp:', nDataTrExp, 'nDataTrAgent:', nDataTrAgent, 'nDataTs:', int(nDataTs),\
              'inputLen:', self.inputLenTime, 'outputLen:', self.outputLenTime)
        
        noiseAmpRatioList = [5e-2, 7.5e-2, 0.1, 0.2, 0.5, 1, 2, 5]
        
        print("----White-box attack performance (Expert)----")
        print('[ampRatio, Acc.]:', end=' ')
        lineBreakCount = 0
        for noiseAmpRatio in noiseAmpRatioList:
            correct = 0.
            for tsData in tsLoader:
                tsObswoPad = tsData['obs'][:, self.padLen:, :]
                pred, label = getPredsGAIL(tsObswoPad, tsData['FGM'], tsData['label'],\
                                              HARTargetNet, noiseAmpRatio)
                # for pred, label in zip(pred_l, label_l):
                correct += (pred == label)
            print('[{0}, {1:.3f}]'.format(noiseAmpRatio, correct/nDataTs), end=' ')
            lineBreakCount += 1
            if lineBreakCount == 8 and lineBreakCount != len(noiseAmpRatioList):
                print('')
                lineBreakCount = -1
        if lineBreakCount != 0:
            print('')
        
        print('----Random noise attack performance----')
        print('[ampRatio, Acc.]:', end=' ')
        lineBreakCount = 0
        for noiseAmpRatio in noiseAmpRatioList:
            correct = 0.
            for trAgentData in tsLoader:
                tsObswoPad = tsData['obs'][:, self.padLen:, :]
                noiseData = torch.randn(tsObswoPad.shape).to(self.device)
                pred, label = getPredsGAIL(tsObswoPad, noiseData, tsData['label'],\
                                              HARTargetNet, noiseAmpRatio)
                correct += (pred == label)
                # accuracyList.append(correct/nData)
            print('[{0}, {1:.3f}]'.format(noiseAmpRatio, correct/nDataTs), end=' ')
            lineBreakCount += 1
            if lineBreakCount == 8 and lineBreakCount != len(noiseAmpRatioList):
                print('')
                lineBreakCount = -1
        if lineBreakCount != 0:
            print('')

        print("GAIL training starts!")
        bestAcc = [1.001, 1.001]
        accHistory = np.zeros((num_iters, len(noiseAmpRatioList)))
        scoreVHistory = np.zeros((num_iters, 3)) # 0:agentScore, 1:expScore, 2:v
        for iIter in range(num_iters):
            if lineBreakCount != 0 and iIter!= 0:
                print('')
            print('It {}'.format(iIter), end=' ')

            obs = []
            acts = []
            rets = []
            advs = []
            gms = []
            agentScores = []

            correct = [0. for _ in noiseAmpRatioList]
            for trAgentData in trAgentLoader:
                seqLength = trAgentData['obs'].shape[1] - self.padLen
                obsData = torch.Tensor().to(self.device)
                for inputIndex in range(self.inputLenTime):
                    obsData = torch.cat((obsData, trAgentData['obs']\
                            [:, self.padLen - self.inputLenTime + inputIndex - self.delayLen + 1:\
                            self.padLen - self.inputLenTime + inputIndex - self.delayLen + seqLength + 1, :]), dim=2)
                # obsData = torch.cat((obsData, trAgentData['obs'][:, padLen:, :]), dim=2)
                obsDataSq = torch.squeeze(obsData, 0)
                actsDataSq = self.act(obsDataSq)
                actsData = torch.unsqueeze(actsDataSq, 0)

                obs.append(obsDataSq)
                acts.append(actsDataSq)
                agentScores.append(self.d.get_logits(obsDataSq, actsDataSq))

                gmsData = torch.pow(gae_gamma, torch.arange(seqLength)).to(self.device)
                lmbsData = torch.pow(gae_lambda, torch.arange(seqLength)).to(self.device)
                costsData = (-1) * torch.log(self.d(obsDataSq, actsDataSq)).squeeze().detach()
                discCostsData = gmsData * costsData
                discRetsData = torch.flip(torch.flip(\
                        discCostsData.to(self.device), dims=[0]).cumsum(dim=0), dims=[0])
                retsData = discRetsData / gmsData

                self.v.eval()
                currVals = self.v(obsDataSq).detach()
                nextVals = torch.cat((self.v(obsDataSq)[1:], torch.zeros(1, 1, device=self.device)))
                deltasData = costsData.unsqueeze(-1) + gae_gamma * nextVals - currVals
                advsData = FloatTensor([
                        ((gmsData * lmbsData)[:seqLength - j].unsqueeze(-1) * deltasData[j:]).sum()
                        for j in range(seqLength)]).to(self.device)
                
                gms.append(gmsData)
                rets.append(retsData)
                advs.append(advsData)
                # print(currVals.shape, nextVals.shape, deltasData.shape, advsData.shape)

                for noiseAmpIndex, noiseAmpRatio in enumerate(noiseAmpRatioList):
                    pred, label = getPredsGAIL(trAgentData['obs'][:, self.padLen:,:],\
                                            actsData,\
                                            trAgentData['label'],\
                                            HARTargetNet,\
                                            noiseAmpRatio)
                    # print(pred_l, label_l)
                    # for pred, label in zip(pred_l, label_l):
                    correct[noiseAmpIndex] += (pred == label)
                    # print('[{0}, {1:.3f}]'.format(noiseAmpRatio, correct/nDataTrAgent), end=' ')

            if normalize_advantage:
                advFlatten = torch.Tensor().to(self.device)
                for adv in advs:
                    advFlatten = torch.cat((advFlatten, adv))

                advFlatten = (advFlatten - advFlatten.mean()) / advFlatten.std()
                cumIndex = 0
                
                advsNorm = []
                for adv in advs:
                    advsNorm.append(advFlatten[cumIndex:cumIndex+len(adv)])
                    cumIndex += len(adv)
                advs = advsNorm

            self.d.train()
            expObs = []
            expActs = []
            expScores = []
            for trExpData in trExpLoader:
                seqLength = trExpData['obs'].shape[1] - self.padLen
                expObsData = torch.Tensor().to(self.device)
                for inputIndex in range(self.inputLenTime):
                    expObsData = torch.cat((expObsData, trExpData['obs']\
                            [:, self.padLen - self.inputLenTime + inputIndex - self.delayLen + 1:\
                            self.padLen - self.inputLenTime + inputIndex - self.delayLen + seqLength + 1, :]), dim=2)
                # expObsData = torch.cat((expObsData, trExpData['obs'][:, padLen:, :]), dim=2)
                expObsDataSq = torch.squeeze(expObsData, 0)
                expActsDataSq = self.act(expObsDataSq)

                expObs.append(expObsDataSq)
                expActs.append(expActsDataSq)
                expScores.append(self.d.get_logits(expObsDataSq, expActsDataSq))
            
            opt_d.zero_grad()
            lossAgent = 0
            meanScoreAgent = []
            for agentScore in agentScores:
                lossAgent += torch.nn.functional.binary_cross_entropy_with_logits(\
                    agentScore, torch.ones_like(agentScore))
                meanScoreAgent.append(agentScore.mean().item())
            lossExp = 0
            meanScoreExp = []
            for expScore in expScores:
                lossExp += torch.nn.functional.binary_cross_entropy_with_logits(\
                    expScore, torch.zeros_like(expScore))
                meanScoreExp.append(expScore.mean().item())
            loss = (lossAgent / len(agentScores)) + (lossExp / len(expScores))
            loss.backward()
            opt_d.step()
            
            print('d_sc: {0:.2f}, {1:.2f}'.format\
                  ((sum(meanScoreAgent) / len(meanScoreAgent)), (sum(meanScoreExp) / len(meanScoreExp))), end=' ')
            scoreVHistory[iIter, 0] = (sum(meanScoreAgent) / len(meanScoreAgent))
            scoreVHistory[iIter, 1] = (sum(meanScoreExp) / len(meanScoreExp))

            del expScores, agentScores

            self.v.train()
            vList = []
            for obsData, retsData in zip(obs, rets):
                old_params = get_flat_params(self.v).detach()
                old_v = self.v(obsData).detach()

                def constraint():
                    return ((old_v - self.v(obsData)) ** 2).mean()

                grad_diff = get_flat_grads(constraint(), self.v)

                def Hv(v):
                    hessian = get_flat_grads(torch.dot(grad_diff, v), self.v).detach()
                    return hessian

                g = get_flat_grads(
                    ((-1) * (self.v(obsData).squeeze() - retsData) ** 2).mean(), self.v).detach()
                vList.append(((-1) * (self.v(obsData).squeeze() - retsData) ** 2).mean().item())

                s = conjugate_gradient(Hv, g).detach()

                Hs = Hv(s).detach()
                alpha = torch.sqrt(2 * eps / torch.dot(s, Hs))
                # print('alpha:', alpha, 's:', s[:10])

                new_params = old_params + alpha * s

                # print('v:', ((-1) * (self.v(obsData).squeeze() - retsData) ** 2).mean())
                set_params(self.v, new_params)
            
            print('v: {0:.1f}'.format(sum(vList)/len(vList)), end=' ')
            scoreVHistory[iIter, 2] = sum(vList)/len(vList)

            # # print(obs.shape, acts.shape, rets.shape, advs.shape, gms.shape)
            self.pi.train()
            for obsData, actsData, advsData, gmsData in zip(obs, acts, advs, gms):
                old_params = get_flat_params(self.pi).detach()
                old_distb = self.pi(obsData)

                def L():
                    distb = self.pi(obsData)
                    return (advsData * torch.exp(distb.log_prob(actsData)
                                - old_distb.log_prob(actsData).detach())).mean()

                def kld():
                    distb = self.pi(obsData)
                    old_mean = old_distb.mean.detach()
                    old_cov = old_distb.covariance_matrix.sum(-1).detach()
                    mean = distb.mean
                    cov = distb.covariance_matrix.sum(-1)
                    return (0.5) * ((old_cov / cov).sum(-1)
                            + (((old_mean - mean) ** 2) / cov).sum(-1)
                            - self.action_dim
                            + torch.log(cov).sum(-1)
                            - torch.log(old_cov).sum(-1)).mean()

                grad_kld_old_param = get_flat_grads(kld(), self.pi)

                def Hv(v):
                    hessian = get_flat_grads(torch.dot(grad_kld_old_param, v), self.pi).detach()
                    return hessian + cg_damping * v

                g = get_flat_grads(L(), self.pi).detach()
                s = conjugate_gradient(Hv, g).detach()
                Hs = Hv(s).detach()
                new_params = rescale_and_linesearch(g, s, Hs, max_kl, L, kld, old_params, self.pi)
                # print('new_params:', new_params.mean(), obs.shape, acts.shape)
            
                disc_causal_entropy = ((-1) * gmsData * self.pi(obsData).log_prob(actsData)).mean()
                grad_disc_causal_entropy = get_flat_grads(disc_causal_entropy, self.pi)
                new_params += lambda_ * grad_disc_causal_entropy
                set_params(self.pi, new_params)

            print('[ampRatio, Acc.]:', end=' ')
            lineBreakCount = 0
            ampSaveCriteria = [0.5, 1] # two criteria
            for noiseAmpIndex, noiseAmpRatio in enumerate(noiseAmpRatioList):
                print('[{0}, {1:.3f}]'.\
                        format(noiseAmpRatio, correct[noiseAmpIndex]/nDataTrAgent), end=' ')
                accHistory[iIter, noiseAmpIndex] = correct[noiseAmpIndex]/nDataTrAgent

                lineBreakCount += 1
                if lineBreakCount == 8 and lineBreakCount != len(noiseAmpRatioList):
                    print('')
                    lineBreakCount = 0

            with open('./savedModels/varGAIL/logs/' + saveFileName + '_accHistory.npy', 'wb') as f:
                np.save(f, accHistory)
            with open('./savedModels/varGAIL/logs/' + saveFileName + '_scoreVHistory.npy', 'wb') as f:
                np.save(f, scoreVHistory)

            compAmpRatio = np.where(np.array(noiseAmpRatioList) == ampSaveCriteria[0])[0][0]
            if correct[compAmpRatio]/nDataTrAgent < bestAcc[0]:
                bestAcc[0] = correct[compAmpRatio]/nDataTrAgent
                compAmpRatio2nd = np.where(np.array(noiseAmpRatioList) == ampSaveCriteria[1])[0][0]
                bestAcc[1] = correct[compAmpRatio2nd]/nDataTrAgent
                
                torch.save(self.pi.state_dict(), './savedModels/varGAIL/' + saveFileName + '_pi.cpkt')
                torch.save(self.v.state_dict(), './savedModels/varGAIL/' + saveFileName + '_v.cpkt')
                torch.save(self.d.state_dict(), './savedModels/varGAIL/' + saveFileName + '_d.cpkt')
                print('model saved!', end=' ')
            elif correct[compAmpRatio]/nDataTrAgent == bestAcc[0]:
                compAmpRatio2nd = np.where(np.array(noiseAmpRatioList) == ampSaveCriteria[1])[0][0]
                if correct[compAmpRatio2nd]/nDataTrAgent < bestAcc[1]:
                    bestAcc[1] = correct[compAmpRatio2nd]/nDataTrAgent

                    torch.save(self.pi.state_dict(), './savedModels/varGAIL/' + saveFileName + '_pi.cpkt')
                    torch.save(self.v.state_dict(), './savedModels/varGAIL/' + saveFileName + '_v.cpkt')
                    torch.save(self.d.state_dict(), './savedModels/varGAIL/' + saveFileName + '_d.cpkt')
                    print('model saved!', end=' ')

if __name__ == "__main__":
    main()