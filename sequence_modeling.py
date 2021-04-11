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
        self.maxpool = nn.MaxPool2d((1, 25), stride=(1, 25))

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrents=[self.rnn(input[:,:i,:]) for i in range(1,26)]
        recurrents_inv=[self.rnn(input[:,j:,:]) for j in range(1,26)]

        pre_recurrents=[recurrent for recurrent, _ in recurrents]
        suf_recurrents=[recurrent for recurrent, _ in recurrents_inv]

        combines=[torch.cat((x,y),dim=1) for x,y in zip(pre_recurrents,suf_recurrents)]
        combine=torch.cat(combines,dim=2)
        new_input=self.maxpool(combine)

        recurrent, _ = self.rnn(new_input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output=self.linear(recurrent)

        return output