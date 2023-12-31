import time
import torch

from torch.utils.data import Dataset
import torch.nn as nn
from torch import linalg as LA

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.1)

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
        
        data = torch.tensor(self.features[idx]).float().to(self.device)
        # print(data.shape)
        if self.normalize:
            data = data * torch.numel(data)/ LA.norm(data)
        label = torch.tensor(self.labels[idx]).long().to(self.device)
        return {'input': data, 'label': label}

def getAcc(loader, model, noiseAmpRatio = 0.0, noiseType = 'random'):
    '''
    get accuracy from predictions
    '''
    pred_l,label_l = getPreds(loader, model, noiseAmpRatio=noiseAmpRatio, noiseType=noiseType)
    correct = 0.
    for pred, label in zip(pred_l,label_l):
        correct += (pred==label)

    return correct/len(pred_l)


def getPreds(loader, model, noiseAmpRatio = 0.0, noiseType = 'random', print_time = False):
    # get predictions from network
    device = model.device
    model.eval()
    pred_l   = []
    label_l = [] 
    
    if noiseType == 'FGM':
        model.train()

    start = time.time()
    for batch in loader:
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
        batch['input'] = noiseInputFlatten.view(batch['input'].shape)

        outputs = model(batch['input'])

        pred_l.extend(outputs.detach().max(dim=1).indices.cpu().tolist())
        label_l.extend(batch['label'].cpu().tolist())
        
    if print_time:
        end = time.time()
        print('time per example:', (end-start)/len(label_l))

    return pred_l, label_l