from easydict import EasyDict
import torch

config = EasyDict()

# dataloader jobs number
config.num_workers = 4

# batch_size
config.batch_size = 5
config.max_epoch = 300
config.start_epoch = 0
config.lr = 1e-4
config.cuda = True
config.backbone_name = 'VGG16'


# data args
config.stds = (0.229, 0.224, 0.225)
config.means = (0.485, 0.456, 0.406)


def update_config(config, extra_config):
    for k, v in vars(extra_config).items():
        config[k] = v
    config.device = torch.device('cuda') if config.cuda else torch.device('cpu')


def print_config(config):
    print('==========Options============')
    for k, v in config.items():
        print('{}: {}'.format(k, v))
    print('=============End=============')