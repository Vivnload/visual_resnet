import sys
import cv2
from matplotlib import pyplot as plt
import numpy as np
import argparse
if './pytorch-grad-cam-master' not in sys.path:
    sys.path.append('./pytorch-grad-cam-master')
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import  transforms
import torch
from model import Model
from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate,hierarchical_dataset
import torch.backends.cudnn as cudnn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
aa=None
def hook(module, inputdata, output):
    global aa
    aa=output.data
    # return output.data
def test(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)
    model = torch.nn.DataParallel(model).to(device)

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))
    opt.exp_name = '_'.join(opt.saved_model.split('/')[1:])
    # print(model)
    # return model
    AlignCollate_evaluation = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    eval_data, eval_data_log = hierarchical_dataset(root=opt.eval_data, opt=opt)
    evaluation_loader = torch.utils.data.DataLoader(
        eval_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_evaluation, pin_memory=True)
    # _, accuracy_by_best_model, _, _, _, _, _, _ = validation(
    #     model, criterion, evaluation_loader, converter, opt)

    for i, (image_tensors, labels) in enumerate(evaluation_loader):
        # batch_size = image_tensors.size(0)
        # text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)
        target_layer = model.module.FeatureExtraction.ConvNet.layer4[-1]
        # model.eval()
        # handle =model.module.Transformation.register_forward_hook(hook)
        # model(image_tensors,text_for_pred)
        input_tensor=image_tensors
        # print(labels)
        # print(input_tensor.shape,'input_tensor.shape')
        # handle.remove()
        # print(input_tensor)
        # Create an input tensor image for your model..
        # input_tensor=image_tensors
        # Note: input_tensor can be a batch tensor with several images!
        # print(labels)
        # Construct the CAM object once, and then re-use it on many images:
        cam = EigenCAM(model=model, target_layer=target_layer, use_cuda=opt.use_cuda)

        # If target_category is None, the highest scoring category
        # will be used for every image in the batch.
        # target_category can also be an integer, or a list of different integers
        # for every image in the batch.
        text_for_loss, length_for_loss = converter.encode(labels, batch_max_length=opt.batch_max_length)

        target_category = text_for_loss


        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category)

        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        loader=transforms.ToPILImage()
        sourc_image=loader(image_tensors[0])
        sourc_image=cv2.cvtColor(np.asarray(sourc_image),cv2.COLOR_RGB2BGR)
        # rgb_img=loader(input_tensor[0].cpu())
        # rgb_img.save('rgb_visual.bmp')
        # rgb_img2=cv2.imread('rgb_visual.bmp')
        rgb_img2=np.float32(sourc_image) / 255
        visualization = show_cam_on_image(rgb_img2, grayscale_cam)
        sourc_image = cv2.resize(sourc_image, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        visualization = cv2.resize(visualization, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_CUBIC)
        cat_image=np.vstack((sourc_image,visualization))
        cv2.imwrite('visual_image2/'+str(i)+'_2.bmp',cat_image)
        # cv2.namedWindow("source img", cv2.WINDOW_NORMAL)
        # cv2.namedWindow("input img",cv2.WINDOW_NORMAL)
        # cv2.namedWindow("attention_map",cv2.WINDOW_NORMAL)
        # cv2.imshow("source img", sourc_image)
        # cv2.imshow("input img", cat_image)
        # cv2.imshow("attention_map", visualization)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # plt.imshow(visualization,'gray')
        # plt.show()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_data', required=True, help='path to evaluation dataset')
    parser.add_argument('--benchmark_all_eval', action='store_true', help='evaluate 10 benchmark evaluation datasets')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--use_cuda', action='store_true',help='cuda')

    opt = parser.parse_args()

    # """ vocab / character number configuration """
    # if opt.sensitive:
    #     opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    test(opt)