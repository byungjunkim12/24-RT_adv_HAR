import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTMNet(nn.Module):
    def __init__(self, nClasses, input_size, hidden_size, num_layers, seq_length, device):
        super(LSTMNet, self).__init__()
        self.nClasses = nClasses #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.device = device

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        # self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(hidden_size, nClasses) #fully connected last layer
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self,x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) #hidden state
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) #internal state
        # Propagate input through LSTM
        _, (hn, _) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc(out) #Final Output
        out = self.softmax(out)

        return out