import time
import torch

from torch.utils.data import Dataset
import torch.nn as nn
from torch import linalg as LA

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.1)

class FGMDataset(Dataset):
    def __init__(self, dataDict, device, normalize=True):
        self.device = device
        self.obs = dataDict['obs'].float() # using IQ sample value
        self.FGM = dataDict['FGM'].float()
        self.labels = dataDict['label'].long()
        self.normalize = normalize

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        obs = self.obs[idx].clone().detach().requires_grad_(True)
        # torch.tensor(self.obs[idx], device=self.device)
        if self.normalize:
            obs = obs * torch.numel(obs)/ LA.norm(obs)
        FGM = self.FGM[idx].clone().detach().requires_grad_(True)
        label = self.labels[idx].clone().detach()
        # obs = self.obs[idx]
        # if self.normalize:
        #     obs = obs * torch.numel(obs)/ LA.norm(obs)
        # FGM = self.FGM[idx]
        # label = self.labels[idx]

        return {'obs': obs, 'FGM': FGM, 'label': label}

class CSIDataset(Dataset):
    def __init__(self, dataDict, device, normalize=True):
        self.features = dataDict['input'] # using IQ sample value
        self.labels = dataDict['label']
        self.device = device
        self.normalize = normalize

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        data = torch.tensor(self.features[idx], device=self.device).float()
        if self.normalize:
            data = data * torch.numel(data)/ LA.norm(data)

        label = torch.tensor(self.labels[idx], device=self.device).long()
        return {'input': data, 'label': label}

def getAcc(loader, loaderPadLen, model, noiseAmpRatio = 0.0, noiseType = 'random'):
    '''
    get accuracy from predictions
    '''
    pred_l,label_l = getPreds(loader, loaderPadLen,\
                            model,\
                            noiseAmpRatio=noiseAmpRatio,\
                            noiseType=noiseType)

    correct = 0.
    for pred, label in zip(pred_l,label_l):
        correct += (pred==label)

    return correct/len(pred_l)


def getPreds(loader, loaderPadLen, model, noiseAmpRatio = 0.0, noiseType = 'random', print_time = False):
    # get predictions from network
    device = model.device
    model.eval()
    pred_l   = []
    label_l = [] 
    
    if noiseType == 'FGM':
        model.train()

    start = time.time()
    for batch in loader:
        if loaderPadLen > 0:
            batch['input'] = batch['input'][:, loaderPadLen:, :]
        inputFlatten = batch['input'].view(batch['input'].shape[0], -1)
        noiseAmp = LA.norm(inputFlatten, dim=1) * noiseAmpRatio
        if noiseType == 'random':
            noise = torch.randn(inputFlatten.shape, device=device)
        elif noiseType == 'FGM':
            batch['input'].requires_grad = True
            model.zero_grad()
            loss = nn.CrossEntropyLoss()
            loss = loss(model(batch['input']), batch['label'])
            loss.backward()
            noise = (batch['input'].grad.data).view(batch['input'].shape[0], -1)
        noise = torch.mul(torch.div(noise, LA.norm(noise, dim=1).unsqueeze(1)),\
                            noiseAmp.unsqueeze(1))
        # print(LA.norm(noise, dim=1).mean())
        noiseInputFlatten = inputFlatten + noise
        batchInput = noiseInputFlatten.view(batch['input'].shape)

        outputs = model(batchInput)

        pred_l.extend(outputs.detach().max(dim=1).indices.cpu().tolist())
        label_l.extend(batch['label'].cpu().tolist())
        
    if print_time:
        end = time.time()
        print('time per example:', (end-start)/len(label_l))

    return pred_l, label_l


# def getAccGAIL(loader, model, noiseAmpRatio = 0.0):
#     '''
#     get accuracy from predictions
#     '''
#     pred_l,label_l = getPredsGAIL(loader, model, noiseAmpRatio)

#     correct = 0.
#     for pred, label in zip(pred_l,label_l):
#         correct += (pred==label)

#     return correct/len(pred_l)


def getPredsGAIL(inputBatch, noiseBatch, labelBatch, model, noiseAmpRatio = 0.0, print_time = False):
    # get predictions from network
    device = model.device
    model.eval()
    pred_l   = []
    label_l = [] 
    
    start = time.time()

    inputFlatten = torch.reshape(inputBatch, (inputBatch.shape[0], -1))
    noiseAmp = LA.norm(inputFlatten, dim=1) * noiseAmpRatio
    noiseFlatten = torch.reshape(noiseBatch, (noiseBatch.shape[0], -1))
    noiseNormalized = torch.mul(torch.div(noiseFlatten, LA.norm(noiseFlatten, dim=1).unsqueeze(1)),\
            noiseAmp.unsqueeze(1))
    noise = noiseNormalized.view(inputBatch.shape)

    batchInput = inputBatch + noise

    outputs = model(batchInput)

    pred_l.extend(outputs.detach().max(dim=1).indices.cpu().tolist())
    label_l.extend(labelBatch.cpu().tolist())
        
    if print_time:
        end = time.time()
        print('time per example:', (end-start)/len(label_l))

    return pred_l, label_l
