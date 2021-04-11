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
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)
        self.maxpoll=nn.MaxPool2d((4,1),(4,1))

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input[:,:4,:])  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        recurrent2,_=self.rnn(input[:,4:,:])
        combine_1=torch.cat([recurrent,recurrent2],dim=1)

        recurrent3, _ = self.rnn(input[:, :8, :])  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        recurrent4, _ = self.rnn(input[:, 8:, :])
        combine_2 = torch.cat([recurrent3, recurrent4], dim=1)

        recurrent5, _ = self.rnn(input[:, :12, :])  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        recurrent6, _ = self.rnn(input[:, 12:, :])
        combine_3 = torch.cat([recurrent5, recurrent6], dim=1)

        recurrent7, _ = self.rnn(input[:, :16, :])  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        recurrent8, _ = self.rnn(input[:, 16:, :])
        combine_4 = torch.cat([recurrent7, recurrent8], dim=1)


        combines=torch.cat([combine_1,combine_2,combine_3,combine_4],dim=1)

        combines=self.maxpoll(combines)

        output = self.linear(combines)  # batch_size x T x output_size

        return output


