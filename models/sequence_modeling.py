import torch.nn as nn
import torch
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        self.rnn=Sub_BidirectionalLSTM_cell(input_size,hidden_size,26)
        self.linear = nn.Linear(hidden_size * 2, output_size)
    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        # recurrent5, _ = self.rnn(input[:, :12, :])  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        # recurrent6, _ = self.rnn(input[:, 12:, :])
        # combine_3 = torch.cat([recurrent5, recurrent6], dim=1)
        combine_3=self.rnn(input)
        output = self.linear(combine_3)  # batch_size x T x output_size
        return output

class Sub_BidirectionalLSTM_cell(nn.Module):
    def __init__(self,input_size,hidden_size,max_length):
        super(Sub_BidirectionalLSTM_cell,self).__init__()
        self.input_length=max_length
        self.hidden_size=hidden_size
        # self.rnncell=nn.LSTMCell(input_size,hidden_size)
        self.rnn=nn.LSTM(input_size,hidden_size,batch_first=True,bidirectional=True)
        self.max_pool=nn.MaxPool2d((max_length-1,1),(max_length-1,1))
    def forward(self, input):

        # batch_size=input.size(0)
        # num_steps=self.input_length
        # output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(device)
        # hidden = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device),
        #           torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(device))
        splic_lists=[[i,self.input_length-i] for i in range(1,self.input_length)]
        split_tensor=[]
        for splic_list in splic_lists:
            left_tensor,right_tensor=torch.split(input,splic_list,dim=1)
            split_tensor.append(torch.cat((left_tensor,right_tensor),dim=1))
        combine=torch.cat(split_tensor,dim=1)
        combine=self.max_pool(combine)
        output,_=self.rnn(combine)
        return output



if __name__ == '__main__':
    a=torch.randn(100,26,256)
    model=Sub_BidirectionalLSTM(256,256,256)
    model2=BidirectionalLSTM(256,256,256)
    c=model2(a)
    b = model(a)
    print(b.shape)
    print(c.shape)



