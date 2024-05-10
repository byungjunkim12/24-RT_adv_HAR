import time
import torch

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
        
        return {'obs': obs, 'FGM': FGM, 'label': label}

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
        data = data[self.padLen:, :]
        if self.normalize:
            data = data * torch.numel(data) /\
                (LA.norm(data) * self.nSubC * self.nRX)
        # print(LA.norm(data))

        label = torch.tensor(self.labels[idx], device=self.device).long()
        return {'input': data, 'label': label}


def getAcc(loader, loaderPadLen, modelTarget, modelSurrogate, variableLen=False, noiseAmpRatio = 0.0, noiseType = 'random'):
    '''
    get accuracy from predictions
    '''
    pred_l,label_l = getPreds(loader,loaderPadLen,\
                            modelTarget,\
                            modelSurrogate,\
                            variableLen,\
                            noiseAmpRatio=noiseAmpRatio,\
                            noiseType=noiseType)
    
    # print(pred_l)

    correct = 0.
    for pred, label in zip(pred_l,label_l):
        correct += (pred==label)

    return correct/len(pred_l)


def getPreds(loader, loaderPadLen, modelTarget, modelSurrogate, variableLen=False, noiseAmpRatio = 0.0, noiseType = 'random', print_time = False):
    # get predictions from network
    device = modelTarget.device
    modelTarget.eval()
    pred_l   = []
    label_l = [] 
    
    if noiseType == 'FGM':
        modelSurrogate.train()

    start = time.time()
    for batch in loader:
        if variableLen:
            batchInput, batchLabel = batch
        else:
            batchInput = batch['input']
            batchLabel = batch['label']
        
        batchLabel = batchLabel.to(device)
    
        # if loaderPadLen > 0:
        #     batchInput = batchInput[:, loaderPadLen:, :]
        inputFlatten = batchInput.view(batchInput.shape[0], -1)
        noiseAmp = LA.norm(inputFlatten, dim=1) * noiseAmpRatio
        if noiseType == 'random':
            noise = torch.randn(inputFlatten.shape, device=device)
        elif noiseType == 'FGM':
            batchInput.requires_grad = True
            modelSurrogate.zero_grad()
            loss = nn.CrossEntropyLoss()
            loss = loss(modelSurrogate(batchInput), batchLabel)
            loss.backward()
            noise = (batchInput.grad.data).view(batchInput.shape[0], -1)
        noise = torch.mul(torch.div(noise, LA.norm(noise, dim=1).unsqueeze(1)),\
                            noiseAmp.unsqueeze(1))

        # print(LA.norm(inputFlatten), LA.norm(noise))
        inputNoiseFlatten = inputFlatten + noise
        batchInput = inputNoiseFlatten.view(batchInput.shape)
        # print(LA.norm(inputFlatten).item(), LA.norm(noise).item())

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
        # print(noise)
        obFlatten = torch.flatten(obWoPad)
        noiseAmp = LA.norm(obFlatten) * noiseAmpRatio
        noiseFlatten = torch.flatten(noise)
        noiseNormalized = torch.mul(torch.div(noiseFlatten, LA.norm(noiseFlatten)),\
                noiseAmp)
        noise = noiseNormalized.view(obWoPad.shape)

        obWNoise = obWoPad + noise
        obWNoise = obWoPad.to(device) + noise.to(device)
        # print(LA.norm(obWoPad).item(), LA.norm(noiseFlatten).item(), LA.norm(noise).item())
        outputs = model(obWNoise.unsqueeze(0))

        pred_l.extend(outputs.detach().max(dim=1).indices.cpu().tolist())
        label_l.append(label.cpu())
        
    return pred_l, label_l