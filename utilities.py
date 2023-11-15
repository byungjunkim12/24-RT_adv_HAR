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

def getAcc(loader, model):
    '''
    get accuracy from predictions
    '''
    pred_l,label_l = getPreds(loader, model)

    correct = 0.
    for pred, label in zip(pred_l,label_l):
        correct += (pred==label)

    return correct/len(pred_l)

def getPreds(loader,model,print_time = False):
    '''
    get predictions from network
    '''

    model.eval()
    pred_l   = []
    label_l = [] 

    start = time.time()
    for batch in loader:
        outputs = model(batch['input'])
        # print('outputs:',outputs.shape)
        pred_l.extend(outputs.detach().max(dim=1).indices.cpu().tolist())
        label_l.extend(batch['label'].cpu().tolist())
        
    if print_time:
        end = time.time()
        print('time per example:', (end-start)/len(label_l))

    return pred_l, label_l