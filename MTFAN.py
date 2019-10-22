import torch, torch.nn as nn, torch.nn.functional as F, math
from torch.nn.modules.conv import _ConvNd
from utils import *
from torch.nn.modules.utils import _single, _pair, _triple
import inspect

def conv3x3(in_planes, out_planes, strd=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return MyConv2d(in_planes, out_planes, kernel_size=3,
                     stride=strd, padding=padding, bias=bias)

def convertLayer(m):
    if isinstance(m, MyConv2d):
       m.old_weight = (m.weight.clone()).detach()
       m.old_weight.requires_grad = False
       m.old_weight = m.old_weight.to('cuda')
       m.new_weight = torch.zeros_like(m.old_weight)
       m.new_weight = m.new_weight.to('cuda')
       m.register_parameter('W', torch.nn.Parameter(torch.eye(m.out_channels, m.out_channels)))
       m.__delattr__('weight')
       

def convertBack(m):
    if hasattr(m, 'W'):
        temp = torch.mm(m.W.data, m.old_weight.view(m.old_weight.size(0), -1))
        new_weight = temp.view(temp.size(0), m.old_weight.size(1), m.kernel_size[0], m.kernel_size[1])
        m.register_parameter('weight', torch.nn.Parameter(new_weight))
        m.__delattr__('W')
        m.__delattr__('old_weight')
        m.__delattr__('new_weight')

class MyConv2d(_ConvNd):
    # by default MyConv2d is equal to Conv2d, it converts into the new conv after convertLayer is applied
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        if str(inspect.signature(_ConvNd)).find('padding_mode') > -1:
            super(MyConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                False, _pair(0), groups, bias, padding_mode='zeros')
        else:
            super(MyConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                False, _pair(0), groups, bias)
        
    def forward(self, input):
        if hasattr(self,'weight'):
            return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)    
        else:
            temp = torch.mm(self.W, self.old_weight.view(self.old_weight.size(0), -1))
            self.new_weight = temp.view(temp.size(0), self.old_weight.size(1), self.kernel_size[0], self.kernel_size[1])
            return F.conv2d(input, self.new_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ConvBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, int(out_planes / 2))
        self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4))
        self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4))

        if in_planes != out_planes:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_planes),
                nn.ReLU(True),
                nn.Conv2d(in_planes, out_planes,
                          kernel_size=1, stride=1, bias=False),
            )
        else:
            self.downsample = None

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = F.relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = F.relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = F.relu(out3, True)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3

class HourGlass(nn.Module):
    def __init__(self, num_modules, depth, num_features):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features

        self._generate_network(self.depth)

    def _generate_network(self, level):
        self.add_module('b1_' + str(level), ConvBlock(self.features, self.features))

        self.add_module('b2_' + str(level), ConvBlock(self.features, self.features))

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.add_module('b2_plus_' + str(level), ConvBlock(self.features, self.features))

        self.add_module('b3_' + str(level), ConvBlock(self.features, self.features))

    def _forward(self, level, inp):
        # Upper branch
        up1 = inp
        up1 = self._modules['b1_' + str(level)](up1)

        # Lower branch
        low1 = F.avg_pool2d(inp, 2, stride=2)
        low1 = self._modules['b2_' + str(level)](low1)

        if level > 1:
            low2 = self._forward(level - 1, low1)
        else:
            low2 = low1
            low2 = self._modules['b2_plus_' + str(level)](low2)

        low3 = low2
        low3 = self._modules['b3_' + str(level)](low3)

        up2 = F.interpolate(low3, scale_factor=2, mode='nearest')

        return up1 + up2

    def forward(self, x):
        return self._forward(self.depth, x)


class FAN(nn.Module):

    def __init__(self, num_modules=1, n_points=66):
        super(FAN, self).__init__()
        self.num_modules = num_modules

        # Base part
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 128)
        self.conv4 = ConvBlock(128, 256)

        # Stacking part
        for hg_module in range(self.num_modules):
            self.add_module('m' + str(hg_module), HourGlass(1, 4, 256))
            self.add_module('top_m_' + str(hg_module), ConvBlock(256, 256))
            self.add_module('conv_last' + str(hg_module),
                            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
            self.add_module('bn_end' + str(hg_module), nn.BatchNorm2d(256))
            self.add_module('l_' + str(hg_module), nn.Conv2d(256,
                                                            n_points, kernel_size=1, stride=1, padding=0))

            if hg_module < self.num_modules - 1:
                self.add_module(
                    'bl' + str(hg_module), nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0))
                self.add_module('al' + str(hg_module), nn.Conv2d(n_points,
                                                                 256, kernel_size=1, stride=1, padding=0))


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), True)
        x = F.avg_pool2d(self.conv2(x), 2, stride=2)
        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        outputs = []
        for i in range(self.num_modules):
            hg = self._modules['m' + str(i)](previous)

            ll = hg
            ll = self._modules['top_m_' + str(i)](ll)

            ll = F.relu(self._modules['bn_end' + str(i)]
                        (self._modules['conv_last' + str(i)](ll)), True)

            # Predict heatmaps
            tmp_out = self._modules['l_' + str(i)](ll)
            outputs.append(tmp_out)

            if i < self.num_modules - 1:
                ll = self._modules['bl' + str(i)](ll)
                tmp_out_ = self._modules['al' + str(i)](tmp_out)
                previous = previous + ll + tmp_out_

        out = outputs[-1]       
        return out

class GeoDistill(nn.Module):

    def __init__(self, sigma=0.5, temperature=0.1 , out_res=32):
        super(GeoDistill,self).__init__()
        self.softargmax = SoftArgmax2D(softmax_temp=temperature)
        self.heatmap = HeatMap(out_res=out_res, sigma=sigma)

    def forward(self,x):
        pts = self.softargmax(x)
        out = self.heatmap(pts)
        return out, pts
