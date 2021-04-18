import torch.nn as nn

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import ResNet_FeatureExtractor
from modules.SUb_seq_and_predic import Sub_seq
from modules.SUb_decode import SUBDecoder
import torch


class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt

        """ Transformation """
        self.Transformation = TPS_SpatialTransformerNetwork(
            F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW),
            I_channel_num=opt.input_channel)

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        # self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)

        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1
        """ Visual Features Refinement """

        """  Contextual Refinement Block """
        self.seqmodel=Sub_seq(self.FeatureExtraction_output, opt.hidden_size,opt.hidden_size)
        self.SequenceModeling_output = opt.hidden_size
        self.decode = SUBDecoder(self.SequenceModeling_output, opt.hidden_size,
                                                             opt.num_class)
    def forward(self, input, attn_text, is_train=True):
        """ Transformation stage """
        # print(input.shape,'input_star_shape')
        input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)

        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        # Selective Contextual Refinement Block
        visual_feature_splits=torch.chunk(visual_feature,visual_feature.size(1),dim=1)
        visual_feature_list=[]
        for visual_feature_split in visual_feature_splits:
            seq_visual_feature=self.seqmodel(visual_feature_split)
            visual_feature_list.append(seq_visual_feature)
        seq_visual_features=torch.cat(visual_feature_list,dim=1)

        pres = self.decode(seq_visual_features,  attn_text, is_train,self.opt.batch_max_length)


        return pres
if __name__ == '__main__':
    a=torch.randn(80,1,32,100)
    # model=Sub_lstm(256,256,256)
    # b=model(a)