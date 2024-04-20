import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence


class LSTMNet(nn.Module):
    def __init__(self, nClasses, input_size, bidirectional, hidden_size, num_layers, seq_length, device):
        super(LSTMNet, self).__init__()
        self.nClasses = nClasses #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.bidirectional = bidirectional #flag indicating whether LSTM is bidirectional
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length
        self.device = device

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True,
                          bidirectional=bidirectional, device=self.device) #lstm
        # self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        if bidirectional:
            self.fc = nn.Linear(2*hidden_size*seq_length, nClasses, device=self.device) #fully connected last layer
        else:
            self.fc = nn.Linear(hidden_size*seq_length, nClasses, device=self.device) #fully connected last layer
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self,x):
        if self.bidirectional:
            h_0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(self.device) #hidden state
            c_0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(self.device) #internal state
        else:
            h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) #hidden state
            c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) #internal state
            
        # Propagate input through LSTM
        # _, (hn, _) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        # if self.bidirectional:
        #     hn = hn.view(-1, 2*self.hidden_size) #reshaping the data for Dense layer next
        # else:
        #     hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        # # print(hn.shape)
        # out = self.relu(hn)

        # Propagate input through LSTM
        hi, (hn, _) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hi = hi.reshape(hi.shape[0], -1) #reshaping the data for Dense layer next
        out = self.relu(hi)

        # if self.bidirectional:
        #     hi = hi.reshape(hi.shape[0], -1) #reshaping the data for Dense layer next
        #     out = self.relu(hi)
        # else:
        #     hn = hn.view(-1, self.hidden_size) #reshapinxg the data for Dense layer next
        #     out = self.relu(hn)

        out = self.fc(out) #Final Output
        out = self.softmax(out)

        return out
    

class VariableLSTMNet(nn.Module):
    def __init__(self, nClasses, input_size, bidirectional, hidden_size, num_layers, device):
        super(VariableLSTMNet, self).__init__()
        self.nClasses = nClasses #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.bidirectional = bidirectional #flag indicating whether LSTM is bidirectional
        self.hidden_size = hidden_size #hidden state
        self.device = device

        self.lstm = nn.LSTM(input_size=input_size,\
                        hidden_size=hidden_size,\
                        num_layers=num_layers,\
                        batch_first=True,
                        bidirectional=bidirectional,\
                        device=self.device) #lstm
        # self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        if bidirectional:
            self.fc = nn.Linear(2*hidden_size, nClasses, device=self.device) #fully connected last layer
        else:
            self.fc = nn.Linear(hidden_size, nClasses, device=self.device) #fully connected last layer
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self,x):
        if self.bidirectional:
            h_0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(self.device) #hidden state
            c_0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to(self.device) #internal state
        else:
            h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) #hidden state
            c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device) #internal state
            
        # Propagate input through LSTM
        hi, (hn, _) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        # print(hi.shape)
        # print(hn[:, 0, :20])
        if self.bidirectional:
            hn = hn.view(-1, 2*self.hidden_size) #reshaping the data for Dense layer next
        else:
            hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        # print(hn.shape)
        out = self.relu(hn)

        # Propagate input through LSTM
        # hi, (hn, _) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        # hi = hi.reshape(hi.shape[0], -1) #reshaping the data for Dense layer next
        # out = self.relu(hi)

        # if self.bidirectional:
        #     hn = hn.reshape(-1, self.hidden_size*2) #reshaping the data for Dense layer next
        #     out = self.relu(hn)
        # else:
        #     hn = hn.view(-1, self.hidden_size) #reshapinxg the data for Dense layer next
        #     out = self.relu(hn)

        out = self.fc(out) #Final Output
        out = self.softmax(out)

        return out