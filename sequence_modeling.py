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

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        prefix_recurrents = [self.rnn(input[:, :i, :]) for i in range(1, 26)]
        pre_recurrents = [recurrent for recurrent, _ in prefix_recurrents]
        suffix_recurrents = [self.rnn(input[:, j:, :]) for j in range(1, 26)]
        suf_recurrents = [recurrent for recurrent, _ in suffix_recurrents]

        prefix_hpi_recurrents_left = [recurrent[:, :, :256] for recurrent in pre_recurrents]
        prefix_hpi_recurrents_right = [recurrent[:, :, 256:] for recurrent in pre_recurrents]

        suffix_hsi_recurrents_left = [recurrent[:, :, :256] for recurrent in suf_recurrents]
        suffix_hsi_recurrents_right = [recurrent[:, :, 256:] for recurrent in suf_recurrents]

        ms = [nn.MaxPool2d((i, 256), stride=256) for i in range(1, 26)]

        combine_left = [torch.cat(x, y, dim=2) for x, y in zip(prefix_hpi_recurrents_left, suffix_hsi_recurrents_left)]
        combine_right = [torch.cat(x, y, dim=2) for x, y in
                         zip(prefix_hpi_recurrents_right, suffix_hsi_recurrents_right)]

        left_recurrents_max_polling = [m(recurrent) for recurrent, m in zip(combine_left, ms)]
        right_recurrents_max_polling = [m(recurrent) for recurrent, m in zip(combine_right, ms)]

        # prefix_hpi_recurrents=[recurrent[:,:,:256]  for recurrent,_ in prefix_recurrents]
        # prefix_hsi_recurrents=[recurrent[:,:,256:]  for recurrent,_ in prefix_recurrents]
        # suffix_recurrents = [self.rnn(input[:,j:,:]) for j in range(1,26)]
        # suffix_hpi_recurrents=[recurrent[:,:,:256]  for recurrent,_ in suffix_recurrents]
        # suffix_hsi_recurrents=[recurrent[:,:,256:]  for recurrent,_ in suffix_recurrents]

        # combine_prefix=[torch.cat(x,y,dim=1) for x,y in zip(prefix_hpi_recurrents,prefix_hsi_recurrents)]
        # combine_suffix=[torch.cat(x,y,dim=1) for x,y in zip(suffix_hpi_recurrents,suffix_hsi_recurrents)]

        # max_prefix=[torch.max(prefix,dim=1) for prefix in combine_prefix]
        # max_suffix=[torch.max(prefix,dim=1) for prefix in combine_suffix]

        # prefix_hsi_recurrents=[recurrent[:,:,256:] for i in range(1,26) for recurrent in self.rnn(input[:, :i, :])]
        # suffix_hsi_recurrents=[recurrent[:,:,256:] for j in range(1,26) for recurrent in self.rnn(input[:,j:,:])]
        # suffix_hpi_recurrents=[recurrent[:,:,:256]  for j in range(1,26) for recurrent in self.rnn(input[:,j:,:])]

        # recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        recurrent2, _ = self.rnn2(input[:, :, :256])  # batch_size x T x input_size//2 -> batch_size x T x (hidden_size)
        recurrent3, _ = self.rnn2(input[:, :, 256:])  # batch_size x T x input_size//2 -> batch_size x T x (hidden_size)
        recurrent4 = torch.cat([recurrent2, recurrent3], dim=2)
        # print(recurrent.shape)
        # print(recurrent2.shape)
        # print(recurrent3.shape)
        # print(recurrent4.shape)
        output = self.linear(recurrent4)  # batch_size x T x output_size

        return output
