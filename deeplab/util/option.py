import argparse
import torch
import os
import torch.backends.cudnn as cudnn

from datetime import datetime

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def arg2str(args):
    args_dict = vars(args)
    option_str = datetime.now().strftime('%b%d_%H-%M-%S') + '\n'

    for k, v in sorted(args_dict.items()):
        option_str += ('{}: {}\n'.format(str(k), str(v)))

    return option_str

class BaseOptions(object):

    def __init__(self):

        self.parser = argparse.ArgumentParser()

        # basic opts
        self.parser.add_argument('--exp_name', default='test',type=str, help='Experiment name')
        self.parser.add_argument('--net', default='depict', type=str, choices=['depict', 'depict+', 'depict_ten', 'resnet'], help='Network name')
        self.parser.add_argument('--dataset', default='MNIST_full', type=str, choices=['MNIST_full', 'CIFAR10', 'USPS', 'FASHION_MNIST', 'CMU-PIE', 'USPS2', 'FRGC', 'MNIST_test'], help='Dataset name')
        self.parser.add_argument('--resume', default=True, type=str, help='Path to target resume checkpoint')
        self.parser.add_argument('--num_workers', default=12, type=int, help='Number of workers used in dataloading')
        self.parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
        self.parser.add_argument('--save_dir', default='/home/weijia.wu/workspace/Kaggle/Word_segementation/deeplab/deeplab_me/save_model', help='Path to save checkpoint models')
        self.parser.add_argument('--vis_dir', default='/home/weijia.wu/workspace/Kaggle/Word_segementation/deeplab/deeplab_me/Demo/', help='Path to save visualization images')
        self.parser.add_argument('--soft_ce', default=True, type=str2bool, help='Use SoftCrossEntropyLoss')
        self.parser.add_argument('--input_channel', default=1, type=int, help='number of input channels' )
        self.parser.add_argument('--pretrain', default=False, type=str2bool, help='Pretrained AutoEncoder model')
        self.parser.add_argument('--gt_dir',
                                 default='/data/data_weijia/Kaggle/yibao_competition/seg_data_02_01/number_segment_1/label',
                                 help='Path to save visualization images')

        # train opts
        self.parser.add_argument('--start_iter', default=0, type=int, help='Begin counting iterations starting from this value (should be used with resume)')
        self.parser.add_argument('--max_epoch', default=250, type=int, help='Max epochs')
        self.parser.add_argument('--max_iters', default=50000, type=int, help='Number of training iterations')
        self.parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
        self.parser.add_argument('--lr_adjust', default='fix', choices=['fix', 'poly'], type=str, help='Learning Rate Adjust Strategy')
        self.parser.add_argument('--stepvalues', default=[], nargs='+', type=int, help='# of iter to change lr')
        self.parser.add_argument('--weight_decay', '--wd', default=0., type=float, help='Weight decay for SGD')
        self.parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD lr')
        self.parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
        self.parser.add_argument('--batch_size', default=12, type=int, help='Batch size for training')
        self.parser.add_argument('--optim', default='SGD', type=str, choices=['SGD', 'Adam'], help='Optimizer')
        self.parser.add_argument('--evaluation_freq', default=1, type=int, help='compute index every # epoch')
        self.parser.add_argument('--display_freq', default=200, type=int, help='display training metrics every # iterations')
        self.parser.add_argument('--save_freq', default=10, type=int, help='save weights every # epoch')

        self.parser.add_argument('--confi_thresh', default=0.5, type=int, help='save weights every # epoch')
        self.parser.add_argument('--confi_tcl_thresh', default=0.4, type=int, help='save weights every # epoch')

        # test args
        self.parser.add_argument('--checkepoch', default=0, type=int, help='Load checkpoint number')






    def parse(self, fixed=None):

        if fixed is not None:
            args = self.parser.parse_args(fixed)
        else:
            args = self.parser.parse_args()

        return args

    def initialize(self, fixed=None):

        # Parse options
        self.args = self.parse(fixed)

        # Setting default torch Tensor type
        if self.args.cuda and torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            cudnn.benchmark = True
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        # Create weights saving directory
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)

        # Create weights saving directory of target model
        model_save_path = os.path.join(self.args.save_dir, self.args.exp_name)

        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)

        return self.args

    def update(self, args, extra_options):

        for k, v in extra_options.items():
            setattr(args, k, v)
