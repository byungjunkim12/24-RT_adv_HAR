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
    def __init__(self, dataDict, device, normalize=False, nSubC=1, nRX=1):
        self.features = dataDict['input'] # using IQ sample value
        self.labels = dataDict['label']
        self.device = device
        self.normalize = normalize
        self.nSubC = nSubC
        self.nRX = nRX

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        data = torch.tensor(self.features[idx], device=self.device).float()
        if self.normalize:
            data = data * torch.numel(data) /\
                (LA.norm(data) * self.nSubC * self.nRX)
        # print(LA.norm(data))

        label = torch.tensor(self.labels[idx], device=self.device).long()
        return {'input': data, 'label': label}

class VariableCSIDataset(Dataset):
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


def getAcc(loader, loaderPadLen, model, variableLen=False, noiseAmpRatio = 0.0, noiseType = 'random'):
    '''
    get accuracy from predictions
    '''
    pred_l,label_l = getPreds(loader,loaderPadLen,\
                            model,variableLen,\
                            noiseAmpRatio=noiseAmpRatio,\
                            noiseType=noiseType)
    
    # print(pred_l)

    correct = 0.
    for pred, label in zip(pred_l,label_l):
        correct += (pred==label)

    return correct/len(pred_l)


def getPreds(loader, loaderPadLen, model, variableLen=False, noiseAmpRatio = 0.0, noiseType = 'random', print_time = False):
    # get predictions from network
    device = model.device
    model.eval()
    pred_l   = []
    label_l = [] 
    
    if noiseType == 'FGM':
        model.train()

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
            model.zero_grad()
            loss = nn.CrossEntropyLoss()
            loss = loss(model(batchInput), batchLabel)
            loss.backward()
            noise = (batchInput.grad.data).view(batchInput.shape[0], -1)
        noise = torch.mul(torch.div(noise, LA.norm(noise, dim=1).unsqueeze(1)),\
                            noiseAmp.unsqueeze(1))

        # print(LA.norm(batchInput), LA.norm(noise))
        noiseInputFlatten = inputFlatten + noise
        batchInput = noiseInputFlatten.view(batchInput.shape)

        outputs = model(batchInput)

        pred_l.extend(outputs.detach().max(dim=1).indices.cpu().tolist())
        label_l.extend(batchLabel.cpu().tolist())
        
    if print_time:
        end = time.time()
        print('time per example:', (end-start)/len(label_l))

    return pred_l, label_l


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
