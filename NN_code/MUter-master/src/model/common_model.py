import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data_utils import Dataer
import math
import sys
from utils import total_param

class TestModel(nn.Module):

    def __init__(self, input_shape, classes_num):
        super(TestModel, self).__init__()

        self.fc1 = nn.Linear(in_features=input_shape, out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=classes_num)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x




class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            # nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


##################
###### VGG #######
##################


class VGG(nn.Module):

    def __init__(self, features, num_classes=10, **kwargs):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A']), **kwargs)
    return model


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model


def vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B']), **kwargs)
    return model


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model


def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D']), **kwargs)
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model


def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']), **kwargs)
    return model


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return 




class AllFCModel(nn.Module):

    def __init__(self, dataset):

        super(AllFCModel, self).__init__()

        self.dataset = dataset
        
        if dataset == 'Mnist':
            self.layers_size = [784, 200, 200, 100, 10]
        elif dataset == 'Cifar10':
            self.layers_size = [3072, 1000, 500, 100, 10]
        else:
            raise Exception('No such dataset called {} !'.format(dataset))
            sys.exit()
        
        self.numlayers = len(self.layers_size) - 1
        self.fc = list(range(self.numlayers))

        for layer_index in range(self.numlayers):
            self.fc[layer_index] = nn.Linear(in_features=self.layers_size[layer_index], out_features=self.layers_size[layer_index + 1], bias=False).to('cuda')
        
        self.fc = tuple(self.fc)

        self.W = list(range(self.numlayers))

        for layer_index in range(self.numlayers):
            self.W[layer_index] = self.fc[layer_index].weight
        
        self.W = tuple(self.W)

    def forward(self, x):

        # size of self.numlayers -1 is not consider the output layer, we instead by 'z' 
        a = list(range(self.numlayers - 1)) # a[0] -> the input, a[i](i!=0) -> the vector after the activation like relu, relu(h(i-1))
        h = list(range(self.numlayers - 1)) # h[i] = W[i].t() @ a[i]

        for layer_index in range(self.numlayers - 1):
            if layer_index == 0:
                a[layer_index] = self.fc[layer_index](x)
            else:
                a[layer_index] = self.fc[layer_index](h[layer_index-1])
            
            h[layer_index] = F.relu(a[layer_index])
        
        z = self.fc[-1](h[-1]) # the output layer

        z.retain_grad()

        for item in a:
            item.retain_grad()
        
        for item in h:
            item.retain_grad()

        h = [x] + h
        a = a + [z]

        return z, a, h

    def custom_zero_grad(self):
        self.zero_grad()

        for sub_fc in self.fc:
            sub_fc.zero_grad()

if __name__ == "__main__":
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    # model = vgg16().to(device)
    model = ResNet(ResidualBlock).to(device)
    print(model)
    print('total params : {}'.format(total_param(model)))
    # batchsize = 128
    # dataer = Dataer(dataset_name='Mnist')
    # device = 'cuda'
    # model = AllFCModel('Mnist').to(device)

    # train_loader = dataer.get_loader(batch_size=batchsize, isTrain=True)
    # for image, label in train_loader:
        
    #     image = image.to(device)
    #     label = label.to(device)

    #     z, a, h = model.forward(image.view(image.shape[0], -1))

    #     loss = F.cross_entropy(z, label)

    #     model.custom_zero_grad()

    #     loss.backward()

    #     A_ = []
    #     G_ = []

    #     for layer_index in range(model.numlayers):
    #         G_.append(1/batchsize * a[layer_index].grad.t() @ a[layer_index].grad)
    #         A_.append(1/batchsize * h[layer_index].t() @ h[layer_index])
        

    #     for mat in A_:
    #         print(mat.shape)
        
    #     print('='*100)

    #     for mat in G_:
    #         print(mat.shape)
        

    #     break
    
    