import torchvision.models.vgg as vgg
import torch.nn as nn

class VGG16(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = vgg.vgg11(pretrained=True).features

        self.relu = self.encoder[1]
        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]

    def forward(self, x):
        C1 = self.stage1(x)
        C2 = self.stage2(C1)
        C3 = self.stage3(C2)
        C4 = self.stage4(C3)
        C5 = self.stage5(C4)
        return C1, C2, C3, C4, C5


if __name__ == '__main__':
    import torch
    input = torch.randn((4, 3, 400, 400))
    net = VGG16()
    C1, C2, C3, C4, C5 = net(input)
    print(C1.size())
    print(C2.size())
    print(C3.size())
    print(C4.size())
    print(C5.size())
# torch.Size([4, 64, 200, 200])
# torch.Size([4, 128, 100, 100])
# torch.Size([4, 256, 50, 50])
# torch.Size([4, 512, 25, 25])
# torch.Size([4, 512, 12, 12])
