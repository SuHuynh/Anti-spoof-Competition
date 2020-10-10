import torch
import torch.nn as nn
import math
from torchsummary import summary
from models.bap import BAP
# from bap import BAP
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class EffcientNet_Wsdan(nn.Module):
    def __init__(self, num_classes = 60, M =32, net='efficientnet-b5', pretrained=False):
        super().__init__()
        self.num_classes=num_classes
        self.M=M
        self.net=net
        
        if pretrained==True:
            self.EFN = EfficientNet.from_pretrained(net)
            self.extract_feats = self.EFN.extract_features
            self.num_features = self.EFN._fc.in_features
        else:
            self.EFN = EfficientNet.from_name('efficientnet-b5')
            self.EFN.set_swish()
            self.extract_feats = self.EFN.extract_features
            self.num_features = self.EFN._fc.in_features

        # Attention Maps
        self.attentions = BasicConv2d(self.num_features, self.M, kernel_size=1)

        # Bilinear Attention Pooling
        self.bap = BAP()

        # Classification Layer
        self.fc = nn.Linear(self.M * self.num_features, self.num_classes, bias=False)

    def forward(self, x):
    
        # Feature Maps, Attention Maps and Feature Matrix
        x = self.extract_feats(x)

        atm = self.attentions(x)
        raw_features, pooling_features = self.bap(x, atm)
        x = pooling_features.view(x.size(0), -1)

        x = self.fc(x)
        # print(x.size())
        
        atr_pred = torch.sigmoid(x[:,0:40])
        spoof_type_pred = x[:,40:51]
        illum_pred = x[:,51:56]
        env_pred = x[:,56:59]
        spoof_pred = torch.sigmoid(x[:,59])

        return atm, raw_features, atr_pred, spoof_type_pred, illum_pred, env_pred, spoof_pred

if __name__=='__main__':
    model = EffcientNet_Wsdan().cuda()
    summary(model, (3, 224, 224))
    # model.eval()
    # print(model)
    # input = torch.randn(12,3,224,224).cuda()
    # y = model(input)
    # print(y.size())

