import torch.nn as nn
import torch

class Sub_lstm(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(Sub_lstm, self).__init__()
        self.rnncell=Sub_lstm_cell(input_size,hidden_size)
        self.max_pool=nn.MaxPool2d((25,1),1)
        self.liner=nn.Linear(1,26)
    def forward(self, input):
        Hi_list=[]
        for i in range(1,26):
            Hi=self.rnncell(input,i)
            Hi_list.append(Hi)
        H_sub=torch.cat(Hi_list,dim=1)
        H_sub=self.max_pool(H_sub)
        H_sub=H_sub.permute(0,2,1)
        H_sub=self.liner(H_sub)
        H_sub = H_sub.permute(0, 2, 1)
        return H_sub






class Sub_lstm_cell(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(Sub_lstm_cell, self).__init__()
        self.rnn = Sub_lstm_layer(input_size,hidden_size)
        self.max_pool = nn.MaxPool2d((2, 1), 1)
    def _sequence(self,input,i):
        S_one2i = input[:, :i, :]
        S_i2n = input[:, i:, :]
        input_flip=torch.flip(input,dims=[1])
        S_i2one = input_flip[:, :i, :]
        S_n2i = input[:, i:, :]


        return S_one2i,S_i2n,S_n2i,S_i2one
    def forward(self, input,i):
        S_1_i,S_i_n,S_n_i,S_i_1=self._sequence(input,i)
        hpi_left=self.rnn(S_1_i)
        hsi_left=self.rnn(S_i_n)
        hsi_right=self.rnn(S_n_i)
        hpi_right=self.rnn(S_i_1)
        left = torch.cat((hpi_left, hsi_left), dim=1)
        right = torch.cat((hsi_right, hpi_right), dim=1)
        left=self.max_pool(left)
        right=self.max_pool(right)
        Hi=torch.cat((left,right),dim=2)
        return Hi


class Sub_lstm_layer(nn.Module):
    def __init__(self,input_size,hidden_size):
        super(Sub_lstm_layer,self).__init__()
        self.rnn=nn.LSTM(input_size,hidden_size,batch_first=True)
        self.hidden_size=hidden_size
    def forward(self, input):
        self.rnn.flatten_parameters()
        _,(h,_)=self.rnn(input)
        h0 = h.permute(1, 0, 2)
        # hpi_left=output[0][-1][:self.hidden_size]
        # hsi_right=output[0][0][self.hidden_size:]
        return h0
# class Sub_lstm_layer_2(nn.Module):



if __name__ == '__main__':
    a=torch.randn(100,26,256)
    # print(a[:,1:25:-1,:])
    model=Sub_lstm(256,256)
    b=model(a)
    print(b.shape)

