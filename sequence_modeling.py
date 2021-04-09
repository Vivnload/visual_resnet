import torch.nn as nn
import torch

class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output
class Sub_BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Sub_BidirectionalLSTM, self).__init__()
        # self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.rnn2=nn.LSTM(input_size//2, hidden_size//2, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)
        self.Dropout=nn.Dropout(0.5)
        self.liner2=nn.Linear(output_size, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        # self.rnn.flatten_parameters()
        self.rnn2.flatten_parameters()
        # recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        recurrent2,_=self.rnn2(input[:,:,:256])# batch_size x T x input_size//2 -> batch_size x T x (hidden_size)
        recurrent3,_=self.rnn2(input[:,:,256:])# batch_size x T x input_size//2 -> batch_size x T x (hidden_size)
        recurrent4=torch.cat([recurrent2,recurrent3],dim=2)
        # print(recurrent.shape)
        # print(recurrent2.shape)
        # print(recurrent3.shape)
        # print(recurrent4.shape)
        output = self.linear(recurrent4)  # batch_size x T x output_size

        return output
