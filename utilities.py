import time
import torch

from scipy import signal
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from torch import linalg as LA


def collate_fn(batch):
    inputs = []
    labels = []
    for i in range(len(batch)):
        inputs.append(batch[i]['input'])
        labels.append(batch[i]['label'])

    inputs_padded = pad_sequence([torch.tensor(seq) for seq in inputs],
                                 batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    return inputs_padded, labels

def collate_fn_FGM(batch):
    inputs = []
    labels = []
    FGMs = []
    for i in range(len(batch)):
        inputs.append(batch[i]['input'])
        FGMs.append(batch[i]['FGM'])
        labels.append(batch[i]['label'])

    inputs_padded = pad_sequence([torch.tensor(seq) for seq in inputs],
                                 batch_first=True, padding_value=0)
    FGM_padded = pad_sequence([torch.tensor(seq_FGM) for seq_FGM in FGMs],
                                 batch_first=True, padding_value=0)

    labels = torch.tensor(labels)
    return inputs_padded, labels, FGM_padded


def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.1)

class FGMDataset(Dataset):
    def __init__(self, dataDict, device, normalize=True, nSubC=30, nRX=3, padLen=0, noiseAmpRatio=0.0):
        self.device = device
        self.obs = dataDict['obs'] # using IQ sample value
        self.FGM = dataDict['FGM']
        self.labels = dataDict['label'].long()
        self.normalize = normalize
        self.nSubC = nSubC
        self.nRX = nRX
        self.noiseAmpRatio = noiseAmpRatio

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        obs = self.obs[idx]
        FGM = self.FGM[idx]
        # obs = self.obs[idx].clone().detach().requires_grad_(True)
        # FGM = self.FGM[idx].clone().detach().requires_grad_(True)
        # torch.tensor(self.obs[idx], device=self.device)
        if self.normalize:
            obs = obs.to(self.device) * torch.numel(obs) /\
                (LA.norm(obs).to(self.device) * self.nSubC * self.nRX)
            FGM = FGM.to(self.device) * torch.numel(FGM) /\
                (LA.norm(FGM).to(self.device) * self.nSubC * self.nRX) * self.noiseAmpRatio
        label = self.labels[idx].clone().detach()
        
        return {'obs': obs, 'label': label, 'FGM': FGM}

class CSIDataset(Dataset):
    def __init__(self, dataDict, device, normalize=False, nSubC=1, nRX=1, padLen=0):
        self.features = dataDict['input'] # using IQ sample value
        self.labels = dataDict['label']
        self.device = device
        self.normalize = normalize
        self.nSubC = nSubC
        self.nRX = nRX
        self.padLen = padLen

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        data = torch.tensor(self.features[idx], device=self.device).float()
        # print('what', data.shape)
        data = data[self.padLen:, :]
        if self.normalize:
            data = data * torch.numel(data) /\
                (LA.norm(data) * self.nSubC * self.nRX)
        # print(LA.norm(data))

        label = torch.tensor(self.labels[idx], device=self.device).long()
        # print('getitem', data.shape)
        return {'input': data, 'label': label}


def getAcc(loader, modelTarget, modelSurro, dSampTarget, dSampSurro,\
           variableLen=False, noiseAmpRatio = 0.0, noiseType = 'random'):
    '''
    get accuracy from predictions
    '''
    pred_l,label_l = getPreds(loader,\
                            modelTarget,\
                            modelSurro,\
                            dSampTarget,\
                            dSampSurro,\
                            variableLen,\
                            noiseAmpRatio=noiseAmpRatio,\
                            noiseType=noiseType)
    # print(pred_l)

    correct = 0.
    for pred, label in zip(pred_l,label_l):
        correct += (pred==label)

    return correct/len(pred_l)


def getPreds(loader, modelTarget, modelSurro, dSampTarget, dSampSurro,\
             variableLen=False, noiseAmpRatio=0.0, noiseType='random', slideLen=200, seqLen=1000, print_time = False):
    # get predictions from network
    dSampRatio = int(dSampSurro/dSampTarget)
    device = modelTarget.device
    modelTarget.eval()
    pred_l   = []
    label_l = [] 
    
    if noiseType == 'FGM':
        modelSurro.train()

    if noiseType == 'Univ':
        modelSurro.train()
        nClass = 0
        for batch in loader:
            batchInput, batchLabel = batch
            if batchLabel.item() > nClass:
                nClass = batchLabel.item()
        nClass = nClass+1

        avgNoiseList = [None] * nClass
        longestLenList = [0] * nClass
        for batch in loader:
            batchInput, batchLabel = batch
            if longestLenList[batchLabel.item()] < batchInput.size(1):
                longestLenList[batchLabel.item()] = batchInput.size(1)
        
        for i in range(nClass):
            avgNoiseList[i] = torch.zeros(1, longestLenList[i], batchInput.shape[-1], device=device)

    start = time.time()
    for batch in loader:
        batchInput, batchLabel = batch
        batchLabel = batchLabel.to(device)
        print(batchInput.shape)
        if dSampRatio != 1:
            batchInput = torch.from_numpy(signal.resample_poly(batchInput.cpu(), 1, dSampRatio, axis=1)).to(device)
        print(batchInput.shape)
    
        inputFlatten = batchInput.view(batchInput.shape[0], -1)
        noiseAmp = LA.norm(inputFlatten, dim=1) * noiseAmpRatio
        if noiseType == 'random':
            noise = torch.randn(inputFlatten.shape, device=device)
        elif noiseType == 'FGM' or noiseType == 'Univ':
            batchInput.requires_grad = True
            modelSurro.zero_grad()
            loss = nn.CrossEntropyLoss()
            loss = loss(modelSurro(batchInput), batchLabel)
            loss.backward()
            noise = (batchInput.grad.data).view(batchInput.shape[0], -1)
        
        if noiseType == 'Univ':
            noise = torch.mul(torch.div(noise, LA.norm(noise, dim=1).unsqueeze(1)),\
                noiseAmp.unsqueeze(1))

            noise = noise.view(batchInput.shape)
            noise = torch.from_numpy(signal.resample_poly(\
                noise.cpu(), longestLenList[batchLabel.item()], noise.shape[1], axis=1)).to(device)
            avgNoiseList[batchLabel.item()] += noise

        else:
            if variableLen:
                noise = torch.mul(torch.div(noise, LA.norm(noise, dim=1).unsqueeze(1)),\
                    noiseAmp.unsqueeze(1))
                inputNoiseFlatten = inputFlatten + noise
                batchInput = inputNoiseFlatten.view(batchInput.shape)
                batchTargetLabel = batchLabel
            else:
                # batchInput = inputFlatten.view(batchInput.shape)
                batchInputUpSamp = torch.from_numpy(signal.resample_poly(batchInput.detach().cpu(), dSampRatio, 1, axis=1)).to(device)
                if batchInputUpSamp.shape[1] < seqLen:
                    continue

                batchTargetInput = torch.Tensor().to(device)
                noise = noise.view(batchInput.shape)
                noiseUpSamp = torch.from_numpy(signal.resample_poly(noise.cpu(), dSampRatio, 1, axis=1)).to(device)
                noiseTargetInput = torch.Tensor().to(device)
                for i in range(0, batchInputUpSamp.shape[1], slideLen):
                    if i+seqLen > batchInputUpSamp.shape[1]:
                        break
                    targetInput = batchInputUpSamp[:, i:i+seqLen, :]
                    targetInput = targetInput * torch.numel(targetInput) /\
                        (LA.norm(targetInput) * targetInput.shape[-1])
                    # print(LA.norm(targetInput))
                    batchTargetInput = torch.cat((batchTargetInput, targetInput), dim=0)

                    targetNoise = noiseUpSamp[:, i:i+seqLen, :]
                    targetNoise = targetNoise * torch.numel(targetNoise) /\
                        (LA.norm(targetNoise) * targetNoise.shape[-1]) * noiseAmpRatio
                    noiseTargetInput = torch.cat((noiseTargetInput, targetNoise), dim=0)
                batchTargetLabel = batchLabel * torch.ones(batchTargetInput.shape[0]).to(device)
                batchInput = batchTargetInput + noiseTargetInput

            # print(batchInput.shape, noiseTargetInput.shape, LA.norm(batchInput).item(), LA.norm(noiseTargetInput).item())
            outputs = modelTarget(batchInput)

            pred_l.extend(outputs.detach().max(dim=1).indices.cpu().tolist())
            label_l.extend(batchTargetLabel.cpu().tolist())
    
    if noiseType == 'Univ':
        for batch in loader:
            batchInput, batchLabel = batch
            if variableLen:
                batchLabel = batchLabel.to(device)
                inputFlatten = batchInput.view(batchInput.shape[0], -1)
                noiseAmp = LA.norm(inputFlatten, dim=1) * noiseAmpRatio
                noise = torch.from_numpy(signal.resample_poly(\
                    avgNoiseList[batchLabel.item()].cpu(), batchInput.shape[1], longestLenList[batchLabel.item()], axis=1)).to(device)
                noise = noise.view(batchInput.shape[0], -1)
                noise = torch.mul(torch.div(noise, LA.norm(noise, dim=1).unsqueeze(1)),\
                            noiseAmp.unsqueeze(1))
                inputNoiseFlatten = inputFlatten + noise
                batchInput = inputNoiseFlatten.view(batchInput.shape)

            else:
                batchInputUpSamp = torch.from_numpy(signal.resample_poly(batchInput.detach().cpu(), dSampRatio, 1, axis=1)).to(device)
                if batchInputUpSamp.shape[1] < seqLen:
                    continue    
                
                batchTargetInput = torch.Tensor().to(device)
                inputFlatten = LA.norm(batchInputUpSamp.view(batchInputUpSamp.shape[0], -1), dim=1) * noiseAmpRatio
                noise = torch.from_numpy(signal.resample_poly(\
                    avgNoiseList[batchLabel.item()].cpu(), batchInputUpSamp.shape[1], longestLenList[batchLabel.item()], axis=1)).to(device)
                for i in range(0, batchInputUpSamp.shape[1], slideLen):
                    if i+seqLen > batchInputUpSamp.shape[1]:
                        break
                    targetInput = batchInputUpSamp[:, i:i+seqLen, :]
                    targetInput = targetInput * torch.numel(targetInput) /\
                        (LA.norm(targetInput) * targetInput.shape[-1])
                    batchTargetInput = torch.cat((batchTargetInput, targetInput), dim=0)
                print(batchInput.shape, batchInputUpSamp.shape, batchTargetInput.shape, noise.shape)
    
            outputs = modelTarget(batchTargetInput)

            pred_l.extend(outputs.detach().max(dim=1).indices.cpu().tolist())
            label_l.extend(batchLabel.cpu().tolist())
        
    if print_time:
        end = time.time()
        print('time per example:', (end-start)/len(label_l))

    return pred_l, label_l



def getPredsWin(loader, modelTarget, modelSurro, dSampTarget, dSampSurro,\
             variableLen=False, noiseAmpRatio = 0.0, noiseType = 'random', print_time = False):
    # get predictions from network
    device = modelTarget.device
    modelTarget.eval()
    pred_l   = []
    label_l = [] 
    
    if noiseType == 'FGM':
        modelSurro.train()

    if noiseType == 'Univ':
        modelSurro.train()
        nClass = 0
        for batch in loader:
            if variableLen:
                # batchInput, batchLabel, batchFGM = batch
                batchInput, batchLabel = batch
                if batchLabel.item() > nClass:
                    nClass = batchLabel.item()
            else:
                batchInput = batch['input']
                batchLabel = batch['label']
                if torch.max(batchLabel) > nClass:
                    nClass = torch.max(batchLabel).item()
        nClass = nClass+1

        avgNoiseList = [None] * nClass
        longestLenList = [0] * nClass
        for batch in loader:
            if variableLen:
                # batchInput, batchLabel, batchFGM = batch
                batchInput, batchLabel = batch
                if longestLenList[batchLabel.item()] < batchInput.size(1):
                    longestLenList[batchLabel.item()] = batchInput.size(1)
            else:
                batchInput = batch['input']
                batchLabel = batch['label']
            # nDataList[batchLabel.item()] += 1
        
        if variableLen:
            for i in range(nClass):
                avgNoiseList[i] = torch.zeros(1, longestLenList[i], batchInput.shape[-1], device=device)

    start = time.time()
    for batch in loader:
        if variableLen:
            # batchInput, batchLabel, batchFGM = batch
            batchInput, batchLabel = batch
        else:
            batchInput = batch['input']
            batchLabel = batch['label']
        batchLabel = batchLabel.to(device)
    
        inputFlatten = batchInput.view(batchInput.shape[0], -1)
        noiseAmp = LA.norm(inputFlatten, dim=1) * noiseAmpRatio
        if noiseType == 'random':
            noise = torch.randn(inputFlatten.shape, device=device)
        elif noiseType == 'FGM' or noiseType == 'Univ':
            batchInput.requires_grad = True
            modelSurro.zero_grad()
            loss = nn.CrossEntropyLoss()
            loss = loss(modelSurro(batchInput), batchLabel)
            loss.backward()
            noise = (batchInput.grad.data).view(batchInput.shape[0], -1)
                    
        noise = torch.mul(torch.div(noise, LA.norm(noise, dim=1).unsqueeze(1)),\
                            noiseAmp.unsqueeze(1))
        
        if noiseType == 'Univ':
            noise = noise.view(batchInput.shape)
            noise = torch.from_numpy(signal.resample_poly(\
                noise.cpu(), longestLenList[batchLabel.item()], noise.shape[1], axis=1)).to(device)
            avgNoiseList[batchLabel.item()] += noise
        else:
            inputNoiseFlatten = inputFlatten + noise
            batchInput = inputNoiseFlatten.view(batchInput.shape)
            print(LA.norm(batchInput).item(), LA.norm(inputFlatten).item(), LA.norm(noise).item())
            print(LA.norm(batchInput[0, :, :]).item())
            outputs = modelTarget(batchInput)

            pred_l.extend(outputs.detach().max(dim=1).indices.cpu().tolist())
            label_l.extend(batchLabel.cpu().tolist())
    
    if noiseType == 'Univ':
        for batch in loader:
            if variableLen:
                # batchInput, batchLabel, batchFGM = batch
                batchInput, batchLabel = batch
            else:
                batchInput = batch['input']
                batchLabel = batch['label']
            batchLabel = batchLabel.to(device)
            inputFlatten = batchInput.view(batchInput.shape[0], -1)
            noiseAmp = LA.norm(inputFlatten, dim=1) * noiseAmpRatio
            noise = torch.from_numpy(signal.resample_poly(\
                avgNoiseList[batchLabel.item()].cpu(), batchInput.shape[1], longestLenList[batchLabel.item()], axis=1)).to(device)
            noise = noise.view(batchInput.shape[0], -1)
            
            noise = torch.mul(torch.div(noise, LA.norm(noise, dim=1).unsqueeze(1)),\
                        noiseAmp.unsqueeze(1))
            
            inputNoiseFlatten = inputFlatten + noise
            batchInput = inputNoiseFlatten.view(batchInput.shape)
            outputs = modelTarget(batchInput)

            pred_l.extend(outputs.detach().max(dim=1).indices.cpu().tolist())
            label_l.extend(batchLabel.cpu().tolist())
        
    if print_time:
        end = time.time()
        print('time per example:', (end-start)/len(label_l))

    return pred_l, label_l



def getPredsGAIL(obs, noises, labels, model, noiseAmpRatio=0.0, padLen=0, print_time = False):
    # get predictions from network
    device = model.device
    model.eval()
    pred_l   = []
    label_l = [] 
    
    # start = time.time()
    for ob, noise, label in zip(obs, noises, labels):
        obWoPad = ob[padLen:, :]
        # print('obWoPad:', obWoPad.shape, 'noise:', noise.shape, noiseAmpRatio)
        obFlatten = torch.flatten(obWoPad)
        noiseAmp = LA.norm(obFlatten) * noiseAmpRatio
        noiseFlatten = torch.flatten(noise)
        noiseNormalized = torch.mul(torch.div(noiseFlatten,\
                LA.norm(noiseFlatten)), noiseAmp)
        noise = noiseNormalized.view(obWoPad.shape)

        obWNoise = obWoPad + noise
        obWNoise = obWoPad.to(device) + noise.to(device)
        # print(LA.norm(obWoPad).item(), LA.norm(noiseFlatten).item(), LA.norm(noise).item())
        outputs = model(obWNoise.unsqueeze(0))

        pred_l.extend(outputs.detach().max(dim=1).indices.cpu().tolist())
        label_l.append(label.cpu())
        
    return pred_l, label_l