import torch
import torch.nn as nn
import sys
sys.path.append('./modules')

from SUb_bilstm import Sub_lstm
from SUb_decode import SUBDecoder


class Sub_seq_and_pre(nn.Module):
    def __init__(self, visual_size, hidden_size, num_class):
        super(Sub_seq_and_pre, self).__init__()
        self.sequence_modeling = nn.Sequential(
            Sub_lstm(visual_size, hidden_size, hidden_size),)

        self.sequence_modeling_output = hidden_size

        self.selective_decoder = SUBDecoder(hidden_size, hidden_size, num_class)

    def forward(self, visual_feature, attn_text, is_train, batch_max_length):
        contextual_feature = self.sequence_modeling(visual_feature)
        # D = torch.cat((contextual_feature, visual_feature), 2)
        pres=[]
        for D in contextual_feature:
            block_pred = self.selective_decoder(D, attn_text, is_train, batch_max_length=batch_max_length)
            pres.append(block_pred)

        return  pres
