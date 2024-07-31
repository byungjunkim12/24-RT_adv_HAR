import numpy as np
import torch
from utilities import *
from GAIL.models.nets import PolicyNetwork, ValueNetwork, Discriminator
from GAIL.utils.funcs import get_flat_grads, get_flat_params, set_params, \
    conjugate_gradient, rescale_and_linesearch

if torch.cuda.is_available():
    from torch.cuda import FloatTensor
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    from torch import FloatTensor
from torch.utils.data import DataLoader
from torch.nn import Module


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
        HARTargetNet,
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
        self.HARTargetNet = HARTargetNet
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
    
    def eval(self, trLoader, pi):
        self.pi.load_state_dict(pi.state_dict())
        self.pi.eval()
        
        nDataTs = 0
        for trData in trLoader:
            nDataTs += trData['obs'].shape[0]

        # noiseAmpRatioList = [5e-3, 1e-2, 1e-2, 5e-2, 7.5e-2, 0.1, 0.2, 0.5, 1, 2, 5, 10]
        noiseAmpRatioList = [1e-6, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.5, 1, 2, 5, 10]

        correct = [0. for _ in noiseAmpRatioList]
        lineBreakCount = 0
        for trData in trLoader:
            seqLength = trData['obs'].shape[1] - self.padLen
            obsData = torch.Tensor().to(self.device)
            for inputIndex in range(self.inputLenTime):
                obsData = torch.cat((obsData, trData['obs']\
                            [:, self.padLen - self.inputLenTime + inputIndex - self.delayLen + 1:\
                            self.padLen - self.inputLenTime + inputIndex - self.delayLen + seqLength + 1, :]), dim=2)

            obsDataSq = torch.squeeze(obsData, 0)
            actsDataSq = self.act(obsDataSq)
            actsData = torch.unsqueeze(actsDataSq, 0)

            for noiseAmpIndex, noiseAmpRatio in enumerate(noiseAmpRatioList):
                print(obsData[:, :, -(self.state_dim):].shape, actsData.shape)
                pred, label = getPredsGAIL(obsData[:, :, -(self.state_dim):], actsData, trData['label'],\
                                                self.HARTargetNet, noiseAmpRatio)
                correct[noiseAmpIndex] += (pred == label)

        for noiseAmpIndex, noiseAmpRatio in enumerate(noiseAmpRatioList):
            print('[{0}, {1:.1f}]'.format(noiseAmpRatio, 100*correct[noiseAmpIndex]/nDataTs), end=' ')
        
        lineBreakCount += 1
        if lineBreakCount == 5:
            print('')
            lineBreakCount = 0
        if lineBreakCount != 0:
            print('')
        
        acc = [x/nDataTs for x in correct]
        return acc

    def train(self, trExpLoader, trAgentLoader, tsLoader):
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
        
        # noiseAmpRatioList = [5e-3, 1e-2, 1e-2, 5e-2, 7.5e-2, 0.1, 0.2, 0.5, 1, 2, 5, 10]
        noiseAmpRatioList = [1e-6, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.5, 1, 2, 5, 10]

        # print("----White-box attack performance (Expert)----")
        # print('[ampRatio, Acc.]:', end=' ')
        # lineBreakCount = 0
        # for noiseAmpRatio in noiseAmpRatioList:
        #     correct = 0.
        #     for tsData in tsLoader:
        #         tsObswoPad = tsData['obs'][:, padLen:, :]
        #         pred, label = getPredsGAIL(tsObswoPad, tsData['FGM'], tsData['label'],\
        #                                       HARNetTarget, noiseAmpRatio)
        #         # for pred, label in zip(pred_l, label_l):
        #         correct += (pred == label)
        #     print('[{0}, {1:.3f}]'.format(noiseAmpRatio, correct/nDataTs), end=' ')
        #     lineBreakCount += 1
        #     if lineBreakCount == 5:
        #         print('')
        #         lineBreakCount = -2
        # if lineBreakCount > 0:
        #     print('')

        print("----Black-box attack performance----")
        print('[ampRatio, Acc.]:', end=' ')
        lineBreakCount = 0
        for noiseAmpRatio in noiseAmpRatioList:
            correct = 0.
            for trAgentData in trAgentLoader:
                trAgentObswoPad = trAgentData['obs'][:, self.padLen:, :]
                pred, label = getPredsGAIL(trAgentObswoPad, trAgentData['FGM'], trAgentData['label'],\
                                            self.HARTargetNet, noiseAmpRatio)
                correct += (pred == label)
            for trExpData in trExpLoader:
                trExpObswoPad = trExpData['obs'][:, self.padLen:, :]
                pred, label = getPredsGAIL(trExpObswoPad, trExpData['FGM'], trExpData['label'],\
                                            self.HARTargetNet, noiseAmpRatio)
                correct += (pred == label)
            print('[{0}, {1:.1f}]'.format(noiseAmpRatio, 100*correct/(nDataTrAgent+nDataTrExp)), end=' ')
            lineBreakCount += 1
            if lineBreakCount == 5:
                print('')
                lineBreakCount = -2
        if lineBreakCount > 0:
            print('')

        print('----Random noise attack performance----')
        print('[ampRatio, Acc.]:', end=' ')
        lineBreakCount = 0
        for noiseAmpRatio in noiseAmpRatioList:
            correct = 0.
            for trAgentData in trAgentLoader:
                trAgentObswoPad = trAgentData['obs'][:, self.padLen:, :]
                noiseData = torch.randn(trAgentObswoPad.shape).to(self.device)
                pred, label = getPredsGAIL(trAgentObswoPad, noiseData, trAgentData['label'],\
                                            self.HARTargetNet, noiseAmpRatio)
                correct += (pred == label)
            for trExpData in trExpLoader:
                trExpObswoPad = trExpData['obs'][:, self.padLen:, :]
                noiseData = torch.randn(trExpObswoPad.shape).to(self.device)
                pred, label = getPredsGAIL(trExpObswoPad, noiseData, trExpData['label'],\
                                            self.HARTargetNet, noiseAmpRatio)
                correct += (pred == label)
            print('[{0}, {1:.1f}]'.format(noiseAmpRatio, 100*correct/(nDataTrAgent+nDataTrExp)), end=' ')
            lineBreakCount += 1
            if lineBreakCount == 5:
                print('')
                lineBreakCount = -2
        if lineBreakCount > 0:
            print('')

        print("GAIL training starts!")
        bestAcc = 1.0
        accHistory = np.zeros((num_iters, len(noiseAmpRatioList)))
        for iIter in range(num_iters):
            if lineBreakCount > 0 and iIter!= 0:
                print('')
            print('Iter {}'.format(iIter), end=' ')

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
                                            self.HARTargetNet,\
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
            
            print('scores: {0:.3f}, {1:.3f}'.format\
                  ((sum(meanScoreAgent) / len(meanScoreAgent)), (sum(meanScoreExp) / len(meanScoreExp))), end=' ')

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


            # print(obs.shape, acts.shape, rets.shape, advs.shape, gms.shape)
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
            ampSaveCriterion = 0.1
            for noiseAmpIndex, noiseAmpRatio in enumerate(noiseAmpRatioList):
                print('[{0}, {1:.3f}]'.\
                        format(noiseAmpRatio, correct[noiseAmpIndex]/nDataTrAgent), end=' ')
                accHistory[iIter, noiseAmpIndex] = correct[noiseAmpIndex]/nDataTrAgent

                lineBreakCount += 1
                if lineBreakCount == 5:
                    print('')
                    lineBreakCount = -2

            # with open('./savedModels/GAIL/logs/' + saveFileName + '_accHistory.npy', 'wb') as f:
            #     np.save(f, accHistory)

            compAmpRatio = np.where(np.array(noiseAmpRatioList) == ampSaveCriterion)[0][0]
            if correct[compAmpRatio]/nDataTrAgent < bestAcc:
                bestAcc = correct[compAmpRatio]/nDataTrAgent
                # torch.save(self.pi.state_dict(), './savedModels/varGAIL/' + saveFileName + '_pi.cpkt')
                # torch.save(self.v.state_dict(), './savedModels/varGAIL/' + saveFileName + '_v.cpkt')
                # torch.save(self.d.state_dict(), './savedModels/varGAIL/' + saveFileName + '_d.cpkt')
                # print('model saved!', end=' ')

            # print('log_prob:', self.pi(obs).log_prob(acts).mean().item(), 'grad_disc:', grad_disc_causal_entropy.mean(), 'disc_ent:', disc_causal_entropy.item())
            # # print('pi:', new_params.norm().item(), lambda_, grad_disc_causal_entropy.norm().item())
            # print('iter final v:', self.v.net[0].weight[1, :5].squeeze().detach().cpu().numpy())
            # print('iter final pi:', self.pi.net[0].weight[1, :5].squeeze().detach().cpu().numpy())
        return _