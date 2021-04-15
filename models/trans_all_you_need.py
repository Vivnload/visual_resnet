import torch.nn as nn
import torch
import sys
# print(sys.path)
sys.path.append('./modules')
from encoding import Encoder
from decoding import Decoder



class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self, n_trg_vocab, d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64, dropout=0.1, n_position=200,
            trg_emb_prj_weight_sharing=True, scale_emb_or_prj='prj'):

        super().__init__()

        # self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        #
        # Options here:
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        self.d_model = d_model

        self.encoder = Encoder(n_position=n_position, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=0, dropout=dropout, scale_emb=scale_emb)

        self.decoder = Decoder(n_position=n_position, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            pad_idx=0, dropout=dropout, scale_emb=scale_emb)

        self.trg_word_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        # if trg_emb_prj_weight_sharing:
        #     # Share the weight between target word embedding & last dense layer
        #     self.trg_word_prj.weight = self.decoder.trg_word_emb.weight
        #
        # if emb_src_trg_weight_sharing:
        #     self.encoder.src_word_emb.weight = self.decoder.trg_word_emb.weight


    def forward(self, src_seq):

        src_mask = None
        trg_mask = None
        enc_output, *_ = self.encoder(src_seq, src_mask)
        dec_output, *_ = self.decoder(enc_output, trg_mask, enc_output, src_mask)
        seq_logit = self.trg_word_prj(dec_output)
        if self.scale_prj:
            seq_logit *= self.d_model ** -0.5

        return seq_logit
if __name__ == '__main__':
    a=torch.randn(100,26,512)
    print(a.shape)
    model=Transformer(100)
    b=model(a)
    print(b.shape)
    print(b)