import numpy as np
import torch
import json
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
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--inputJson", type=str, help='input JSON filename')
    parser.add_argument('-c', "--cuda", type=int, default=0, help='cuda device number')
    parser.add_argument('-l', "--load_model", type=bool, default=True, help='load model if model with filename exists')

    args = parser.parse_args()
    inputJsonFileName = args.inputJson
    cudaID = args.cuda
    loadModel = args.load_model

    if cudaID >= 0:
        device = torch.device("cuda:" + str(cudaID))
    else:
        device = torch.device("cpu")

    LSTMModelDir = './savedModels/selected/'
    inputJsonFile = open("./inputJson/GAIL/" + inputJsonFileName + ".json", "r")
    inputJson = json.load(inputJsonFile)
    LSTMModelName = inputJson['LSTMModelName']
    noiseAmpRatio = inputJson['noiseAmpRatio']
    trDataRatio = inputJson['trDataRatio']
    trExpDataRatio = inputJson['trExpDataRatio']
    GAILTrainConfig = inputJson['trainConfig']
    
    torch.set_num_threads(1)

    if cudaID >= 0:
        device = torch.device("cuda:"+str(cudaID))
    else:
        device = torch.device("cpu")

    dataType = LSTMModelName.split('_')[0]
    if dataType == 'survey':
        fs = 1000 # 1 kHz
        nSubC = 30
        nRX = 3
        
        winLen = 1000
        thres = 60
        slideLen = 400
        activities = ['fall', 'pickup', 'run', 'sitdown', 'standup', 'walk']

    LSTMType = LSTMModelName.split('_')[1]
    bidirectional = (LSTMType == 'BLSTM')
    nHidden = int(LSTMModelName.split('_')[3])
    threshold = int(LSTMModelName.split('_')[5])
    nLayer = int(LSTMModelName.split('_')[7])

    # Load the LSTM model
    HARNet = LSTMNet(nClasses=len(activities), input_size=nSubC*nRX, bidirectional=bidirectional,\
                    hidden_size=nHidden, num_layers=1, seq_length=winLen//2, device=device)
    HARNet.load_state_dict(torch.load(LSTMModelDir + LSTMModelName + '.cpkt'))

    # Load dataset labelled with FGM attack
    FGMdatasetDir = '/project/iarpa/wifiHAR/HAR_' + dataType + '/window_FGM/'
    dataDict = {file:[] for file in activities}

    trExpDataset = list()
    trAgentDataset = list()
    tsDataset = list()
    for actInd, activity in enumerate(activities):
        dataDict[activity] = defaultdict(list)

        dataActFileName = FGMdatasetDir + LSTMModelName + '_' + activity + '.pt'
        dataAct = torch.load(dataActFileName)
        dataDict[activity]['obs'] =\
            torch.reshape(torch.squeeze(dataAct[0, :, :]), (-1, winLen//2, nSubC*nRX)).to(device)
        dataDict[activity]['FGM'] = noiseAmpRatio *\
            torch.reshape(torch.squeeze(torch.squeeze(dataAct[1, :, :])),\
                        (-1, winLen//2, nSubC*nRX)).to(device)
        dataDict[activity]['label'] =\
            actInd * torch.ones_like(torch.empty(dataDict[activity]['obs'].shape[0], device=device), dtype=int).to(device)

        datasetAct = FGMDataset(dataDict[activity], device)
        trExpDataset.append(torch.utils.data.Subset(datasetAct,\
                                                    range(int(trDataRatio*trExpDataRatio*len(datasetAct)))))
        trAgentDataset.append(torch.utils.data.Subset(datasetAct,\
                                                    range(int(trDataRatio*trExpDataRatio*len(datasetAct)),\
                                                            int(trDataRatio*len(datasetAct)))))
        tsDataset.append(torch.utils.data.Subset(datasetAct,\
                                                range(int(trDataRatio*len(datasetAct)), len(datasetAct))))

        print('activity:', activity, 'trExpDataset:', len(trExpDataset[-1]),\
            'trAgentDataset:', len(trAgentDataset[-1]), 'tsDataset:', len(tsDataset[-1]))


    trExpLoader = DataLoader(torch.utils.data.ConcatDataset(trExpDataset),\
                        batch_size=20, shuffle=True, generator=torch.Generator(device=device))
    trAgentLoader = DataLoader(torch.utils.data.ConcatDataset(trAgentDataset),\
                        batch_size=20, shuffle=True, generator=torch.Generator(device=device))
    tsLoader = DataLoader(torch.utils.data.ConcatDataset(tsDataset),\
                        batch_size=20, shuffle=True, generator=torch.Generator(device=device))

    print('trExpLoader:', len(trExpLoader), 'trAgentLoader', len(trAgentLoader), 'tsLoader:', len(tsLoader))

    model = GAIL(state_dim=nSubC*nRX, action_dim=nSubC*nRX,\
                discrete=False, device=device, train_config=GAILTrainConfig)
    saveFileName = LSTMModelName + '_' + inputJsonFileName
    if loadModel:
        model.pi.load_state_dict(torch.load('./savedModels/GAIL/' + saveFileName + '_pi.cpkt'))
        model.v.load_state_dict(torch.load('./savedModels/GAIL/' + saveFileName + '_v.cpkt'))
        model.d.load_state_dict(torch.load('./savedModels/GAIL/' + saveFileName + '_d.cpkt'))
        print('model loaded!')

    model.train(HARNet, trExpLoader, trAgentLoader, tsLoader, saveFileName)

class GAIL(Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        discrete,
        device,
        train_config=None
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.discrete = discrete
        self.device = device
        self.train_config = train_config

        self.pi = PolicyNetwork(self.state_dim, self.action_dim, self.discrete, self.device)
        self.v = ValueNetwork(self.state_dim, self.device)
        self.d = Discriminator(self.state_dim, self.action_dim, self.discrete, self.device)

    def get_networks(self):
        return [self.pi, self.v]

    def act(self, state):
        self.pi.eval()
        state = FloatTensor(state)
        action = self.pi(state).sample()
        # action = distb.sample().detach().cpu().numpy()

        return action
    
    def eval(self, trAgentLoader, HARNet, pi, v, d):
        self.pi.load_state_dict(pi.state_dict())
        self.v.load_state_dict(v.state_dict())
        self.d.load_state_dict(d.state_dict())
        
        nDataTrAgent = 0
        for trAgentBatch in trAgentLoader:
            nDataTrAgent += trAgentBatch['obs'].shape[0]

        noiseAmpRatioList = [1e-5, 1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 7.5e-2, 0.1, 0.2, 0.5]
        correct = [0. for _ in noiseAmpRatioList]
        for noiseAmpIndex, noiseAmpRatio in enumerate(noiseAmpRatioList):
            for trAgentBatch in trAgentLoader:
                obsBatchFlatten = trAgentBatch['obs'].transpose(0, 1).reshape(-1, trAgentBatch['obs'].shape[2])
                actBatchFlatten = self.act(obsBatchFlatten)
                actBatch = torch.reshape(actBatchFlatten, ([-1] + list(trAgentBatch['FGM'].shape[1:]))).to(self.device)

                pred_l,label_l = getPredsGAIL(trAgentBatch['obs'], actBatch, trAgentBatch['label'],\
                                              HARNet, noiseAmpRatio)
                for pred, label in zip(pred_l, label_l):
                    correct[noiseAmpIndex] += (pred == label)
            print('[{0}, {1:.3f}]'.format(noiseAmpRatio, correct[noiseAmpIndex]/nDataTrAgent), end=' ')

    def train(self, HARNet, trExpLoader, trAgentLoader, tsLoader, saveFileName, render=False):
        num_iters = self.train_config["num_iters"]
        lambda_ = self.train_config["lambda"]
        gae_gamma = self.train_config["gae_gamma"]
        gae_lambda = self.train_config["gae_lambda"]
        eps = self.train_config["epsilon"]
        max_kl = self.train_config["max_kl"]
        cg_damping = self.train_config["cg_damping"]
        normalize_advantage = self.train_config["normalize_advantage"]

        opt_d = torch.optim.Adam(self.d.parameters(), lr=1e-4)
        # opt_d = torch.optim.Adam(self.d.parameters())


        noiseAmpRatioList = [1e-4, 1e-3, 1e-2, 0.1]
        print('----White-box attack performance (Expert)----')
        lineBreakCount = 0  
        print('[ampRatio, Acc.]:', end=' ')
        nDataTrExp = 0
        for trExpBatch in trExpLoader:
            nDataTrExp += trExpBatch['obs'].shape[0]
        nDataTrAgent = 0
        for trAgentBatch in trAgentLoader:
            nDataTrAgent += trAgentBatch['obs'].shape[0]
        nDataTs = 0
        for tsBatch in tsLoader:
            nDataTs += tsBatch['obs'].shape[0]

        for noiseAmpRatio in noiseAmpRatioList:
            correct = 0.            
            for trAgentBatch in trAgentLoader:
                pred_l,label_l = getPredsGAIL(trAgentBatch['obs'], trAgentBatch['FGM'], trAgentBatch['label'],\
                                              HARNet, noiseAmpRatio)
                for pred, label in zip(pred_l, label_l):
                    correct += (pred == label)
            print('[{0}, {1:.3f}]'.format(noiseAmpRatio, correct/nDataTrAgent), end=' ')

            lineBreakCount += 1
            if lineBreakCount == 4:
                print('')
                lineBreakCount = 0

        if lineBreakCount != 0:
            print('')
        print('----Random noise attack performance----')
        print('[ampRatio, Acc.]:', end=' ')
        for noiseAmpRatio in noiseAmpRatioList:
            correct = 0.
            for trAgentBatch in trAgentLoader:
                noiseBatch = torch.randn(trAgentBatch['obs'].shape).to(self.device)
                pred_l,label_l = getPredsGAIL(trAgentBatch['obs'], noiseBatch, trAgentBatch['label'],\
                                              HARNet, noiseAmpRatio)
                for pred, label in zip(pred_l, label_l):
                    correct += (pred == label)
                # accuracyList.append(correct/nData)
            print('[{0}, {1:.3f}]'.format(noiseAmpRatio, correct/nDataTrAgent), end=' ')
            lineBreakCount += 1
            if lineBreakCount == 4:
                print('')
                lineBreakCount = 0


        if lineBreakCount != 0:
            print('')

        print('nDataTrExp:', nDataTrExp, 'nDataTrAgent:', nDataTrAgent, 'nDataTs:', int(nDataTs))
        bestAcc = 1.0
        for i in range(num_iters):
            if lineBreakCount != 0 and i!= 0:
                print('')
            print('Iter {}'.format(i), end=' ')
            
            obs = []
            acts = []
            rets = []
            advs = []
            gms = []

            correct = [0. for _ in noiseAmpRatioList]
            for tri, trAgentBatch in enumerate(trAgentLoader):
                seqLength = trAgentBatch['obs'].shape[1]
                obsAmp = LA.norm(trAgentBatch['obs'].view(trAgentBatch['obs'].shape[0], -1), dim=1)
                obsBatchFlatten = trAgentBatch['obs'].transpose(0, 1).reshape(-1, trAgentBatch['obs'].shape[2])
                actBatchFlatten = self.act(obsBatchFlatten)
                actBatch = torch.reshape(actBatchFlatten, ([-1] + list(trAgentBatch['FGM'].shape[1:]))).to(self.device)
                
                lineBreakCount = 0
                for noiseAmpIndex, noiseAmpRatio in enumerate(noiseAmpRatioList):
                    pred_l,label_l = getPredsGAIL(trAgentBatch['obs'], actBatch, trAgentBatch['label'],\
                                                    HARNet, noiseAmpRatio)
                    for pred, label in zip(pred_l, label_l):
                        correct[noiseAmpIndex] += (pred == label)
                    
                obs.append(obsBatchFlatten)
                acts.append(actBatchFlatten)

                retsBatch = torch.Tensor().to(self.device)
                advsBatch = torch.Tensor().to(self.device)
                gmsBatch = torch.Tensor().to(self.device)
                for i, trAgentData in enumerate(trAgentBatch['obs']):
                    ep_obs = trAgentData
                    ep_acts = torch.squeeze(actBatch[i, :, :])
                    ep_gms = torch.pow(gae_gamma, torch.arange(seqLength)).to(self.device)
                    ep_lmbs = torch.pow(gae_lambda, torch.arange(seqLength)).to(self.device)
                                        
                    ep_costs = (-1) * torch.log(self.d(ep_obs, ep_acts)).squeeze().detach()
                    ep_disc_costs = ep_gms * ep_costs
                    ep_disc_rets = torch.flip(torch.flip(\
                        ep_disc_costs.to(self.device), dims=[0]).cumsum(dim=0), dims=[0])
                    ep_rets = ep_disc_rets / ep_gms
                    retsBatch = torch.cat((retsBatch, ep_rets), dim=0)

                    self.v.eval()
                    curr_vals = self.v(ep_obs).detach()
                    next_vals = torch.cat(
                        (self.v(ep_obs)[1:], FloatTensor([[0.]]).to(self.device))).detach()
                    ep_deltas = ep_costs.unsqueeze(-1) + gae_gamma * next_vals - curr_vals
                    ep_advs = FloatTensor([
                        ((ep_gms * ep_lmbs)[:seqLength - j].unsqueeze(-1) * ep_deltas[j:]).sum()
                        for j in range(seqLength)]).to(self.device)

                    advsBatch = torch.cat((advsBatch, ep_advs))
                    gmsBatch = torch.cat((gmsBatch, ep_gms))
                
                rets.append(retsBatch)
                advs.append(advsBatch)
                gms.append(gmsBatch)
            
            if normalize_advantage:
                advsFlatten = torch.cat(advs)
                advsFlatten = (advsFlatten - advsFlatten.mean()) / (advsFlatten.std() + 1e-8)
                advs = torch.split(advsFlatten, [len(advsBatch) for advsBatch in advs])
                
            self.d.train()
            expScores = torch.Tensor().to(self.device)
            agentScores = torch.Tensor().to(self.device)
            for trExpBatch in trExpLoader:
                expObsBatch = trExpBatch['obs'].transpose(0, 1).reshape(-1, trExpBatch['obs'].shape[2])
                expActBatch = trExpBatch['FGM'].transpose(0, 1).reshape(-1, trExpBatch['FGM'].shape[2])
                expScores = torch.cat((expScores, self.d.get_logits(expObsBatch, expActBatch)), dim=0)
                
            for agentObsBatch, agentActsBatch in zip(obs, acts):
                agentScores = torch.cat((agentScores, self.d.get_logits(agentObsBatch, agentActsBatch)), dim=0)
            
            opt_d.zero_grad()
            lossExp = torch.nn.functional.binary_cross_entropy_with_logits(\
                expScores, torch.zeros_like(expScores))
            lossAgent = torch.nn.functional.binary_cross_entropy_with_logits(\
                agentScores, torch.ones_like(agentScores))
            loss = lossExp + lossAgent
            loss.backward()
            opt_d.step()

            # print('scores: {0:.3f}, {1:.3f}'.format\
            #       (torch.mean(expScores).item(), torch.mean(agentScores).item()), end=' ')
            print('scores: {0:.3f}, {1:.3f}'.format\
                  (torch.mean(expScores).item(), torch.mean(agentScores).item()))

            
            del expScores
            del agentScores

            self.v.train()
            for obsBatch, actsBatch, retsBatch in zip(obs, acts, rets):
                # print(obsBatch.shape, retsBatch.shape, advsBatch.shape, gmsBatch.shape)
                old_params = get_flat_params(self.v).detach()
                old_vBatch = self.v(obsBatch).detach()
            
                def constraint():
                    return ((old_vBatch - self.v(obsBatch)) ** 2).mean()
            
                grad_diff = get_flat_grads(constraint(), self.v)

                def Hv(v):
                    hessian = get_flat_grads(torch.dot(grad_diff, v), self.v).detach()
                    return hessian

                g = get_flat_grads(\
                    (-1*(self.v(obsBatch).squeeze() - retsBatch) ** 2).mean(), self.v).detach()
                s = conjugate_gradient(Hv, g).detach()
                Hs = Hv(s).detach()
                alpha = torch.sqrt(2 * eps / (torch.dot(s, Hs) + 1e-8)).detach()
                new_params = old_params + alpha * s
                set_params(self.v, new_params)

            print('iter final v:', self.v.net[0].weight[1, :5].squeeze().detach().cpu().numpy())
            
            self.pi.train()
            for obsBatch, actsBatch, advsBatch, gmsBatch in zip(obs, acts, advs, gms):
                old_params = get_flat_params(self.pi).detach()
                old_distb = self.pi(obsBatch)

                def L():
                    distb = self.pi(obsBatch)
                    return (advsBatch * torch.exp(distb.log_prob(actsBatch)\
                                    - old_distb.log_prob(actsBatch).detach())).mean()

                def kld():
                    distb = self.pi(obsBatch)
                    old_mean = old_distb.mean.detach()
                    old_cov = old_distb.covariance_matrix.sum(-1).detach()
                    mean = distb.mean
                    cov = distb.covariance_matrix.sum(-1)
                    return (0.5) * ((old_cov / cov).sum(-1)\
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
                disc_causal_entropy = ((-1) * gmsBatch * self.pi(obsBatch).log_prob(actsBatch)).mean()
                grad_disc_causal_entropy = get_flat_grads(disc_causal_entropy, self.pi)
                new_params += lambda_ * grad_disc_causal_entropy

                set_params(self.pi, new_params)
            
            print('iter final pi:', self.pi.net[0].weight[1, :5].squeeze().detach().cpu().numpy())

            print('[ampRatio, Acc.]:', end=' ')
            lineBreakCount = 0
            ampSaveCriterion = 0.1
            for noiseAmpIndex, noiseAmpRatio in enumerate(noiseAmpRatioList):
                print('[{0}, {1:.3f}]'.\
                        format(noiseAmpRatio, correct[noiseAmpIndex]/nDataTrAgent), end=' ')

                lineBreakCount += 1
                if lineBreakCount == 4:
                    print('')
                    lineBreakCount = 0

            compAmpRatio = np.where(np.array(noiseAmpRatioList) == ampSaveCriterion)[0][0]
            if correct[compAmpRatio]/nDataTrAgent < bestAcc:
                bestAcc = correct[compAmpRatio]/nDataTrAgent
                torch.save(self.pi.state_dict(), './savedModels/GAIL/' + saveFileName + '_pi.cpkt')
                torch.save(self.v.state_dict(), './savedModels/GAIL/' + saveFileName + '_v.cpkt')
                torch.save(self.d.state_dict(), './savedModels/GAIL/' + saveFileName + '_d.cpkt')
                print('model saved!')

if __name__ == "__main__":
    main()