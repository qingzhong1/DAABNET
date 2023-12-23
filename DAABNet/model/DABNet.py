import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

__all__ = ["DABNet"]




class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1,x2], dim=1)
        return out[:,:self.oup,:,:]

###############################################
BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
class SPBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None):
        super(SPBlock, self).__init__()
        midplanes = outplanes
        self.conv1 = nn.Conv2d(inplanes, midplanes, kernel_size=(3, 1), padding=(1, 0), bias=False)
        #self.bn1 = norm_layer(midplanes)
        self.bn1 = BatchNorm2d(midplanes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(inplanes, midplanes, kernel_size=(1, 3), padding=(0, 1), bias=False)
        #self.bn2 = norm_layer(midplanes)
        self.bn2 = BatchNorm2d(midplanes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(midplanes, outplanes, kernel_size=1, bias=True)
        self.pool1 = nn.AdaptiveAvgPool2d((None, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, None))
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        res1=x
        _, _, h, w = x.size()
        x1 = self.pool1(x)
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = x1.expand(-1, -1, h, w)
        #x1 = F.interpolate(x1, (h, w))

        x2 = self.pool2(x)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = x2.expand(-1, -1, h, w)
        #x2 = F.interpolate(x2, (h, w))

        x = self.relu(x1 + x2)
        x = self.conv3(x).sigmoid()
        return x*res1+res1
##############################################################################
class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat
class onlyConv(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(onlyConv, self).__init__()
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                padding=padding, dilation=dilation,
                groups=groups, bias=bias)
        

    def forward(self, x):
        feat = self.conv(x)
        
        return feat

class DetailBranch(nn.Module):

    def __init__(self):
        super(DetailBranch, self).__init__()
        self.S1 = nn.Sequential(
            ConvBNReLU(3, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S2 = nn.Sequential(
            ConvBNReLU(64, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S3 = nn.Sequential(
            ConvBNReLU(64, 128, 3, stride=2),
            ConvBNReLU(128, 128, 3, stride=1),
            onlyConv(128, 128, 3, stride=1),
        )

    def forward(self, x):
        feat = self.S1(x)
        feat = self.S2(feat)
        feat = self.S3(feat)
        return feat
#######################################################################################
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        #return out+identity
        return out


class AttentionRefinementModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        # global average pooling
        x = self.avgpool(input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        # x = self.sigmoid(self.bn(x))
        x = self.sigmoid(x)
        # channels of input and x should be same
        x = torch.mul(input, x)
        return x

#######################################################################################################################################



class Conv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride, padding, dilation=(1, 1), groups=1, bn_acti=False, bias=False):
        super().__init__()

        self.bn_acti = bn_acti

        self.conv = nn.Conv2d(nIn, nOut, kernel_size=kSize,
                              stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)

        if self.bn_acti:
            self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv(input)

        if self.bn_acti:
            output = self.bn_prelu(output)

        return output


class BNPReLU(nn.Module):
    def __init__(self, nIn):
        super().__init__()
        self.bn = nn.BatchNorm2d(nIn, eps=1e-3)
        self.acti = nn.PReLU(nIn)

    def forward(self, input):
        output = self.bn(input)
        output = self.acti(output)

        return output


class DABModule(nn.Module):
    def __init__(self, nIn, d=1, kSize=3, dkSize=3):
        super().__init__()

        self.bn_relu_1 = BNPReLU(nIn)
        self.conv3x3 = Conv(nIn, nIn // 2, kSize, 1, padding=1, bn_acti=True)
        #self.gdconv3x3 = GhostModule(nIn // 2, nIn // 2,relu=True)
        #self.gddconv3x3 = GhostModule(nIn // 2, nIn // 2,relu=True)

        self.dconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1,
                             padding=(1, 0), groups=nIn // 2, bn_acti=True)
        self.dconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1,
                             padding=(0, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv3x1 = Conv(nIn // 2, nIn // 2, (dkSize, 1), 1,
                              padding=(1 * d, 0), dilation=(d, 1), groups=nIn // 2, bn_acti=True)
        self.ddconv1x3 = Conv(nIn // 2, nIn // 2, (1, dkSize), 1,
                              padding=(0, 1 * d), dilation=(1, d), groups=nIn // 2, bn_acti=True)
        self.ca1 = AttentionRefinementModule(nIn // 2, nIn // 2)
        self.ca2 = AttentionRefinementModule(nIn // 2, nIn // 2)
        
        
        self.bn_relu_2 = BNPReLU(nIn // 2)
        self.conv1x1 = Conv(nIn // 2, nIn, 1, 1, padding=0, bn_acti=False)

    def forward(self, input):
        #print(input.shape)
        output = self.bn_relu_1(input)
        output = self.conv3x3(output)
        #br1 = self.gdconv3x3(output)
        #br2 = self.gddconv3x3(output)
        #print(output.shape)

        br1 = self.dconv3x1(output)
        #print(br1.shape)
        
        br2 = self.ddconv3x1(output)
        #print(br2.shape)
        
        output_mid = br1+br2
        br1 = self.ca1(output_mid)
        br2 = self.ca2(output_mid)
        
        br1 = self.dconv1x3(br1)
        #print(br1.shape)
        br2 = self.ddconv1x3(br2)
        #print(br2.shape)

        output = br1 + br2
        output = self.bn_relu_2(output)
        #output = self.ca1(output)
        #output = self.bn_relu_2(output)
        output = self.conv1x1(output)
        #print(output.shape)

        return output + input


class DownSamplingBlock(nn.Module):
    def __init__(self, nIn, nOut):
        super().__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            nConv = nOut - nIn
        else:
            nConv = nOut

        self.conv3x3 = Conv(nIn, nConv, kSize=3, stride=2, padding=1)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.bn_prelu = BNPReLU(nOut)

    def forward(self, input):
        output = self.conv3x3(input)

        if self.nIn < self.nOut:
            max_pool = self.max_pool(input)
            output = torch.cat([output, max_pool], 1)

        output = self.bn_prelu(output)

        return output


class InputInjection(nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, ratio):
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        for pool in self.pool:
            input = pool(input)

        return input
class SegHead(nn.Module):
    def __init__(self, c1, ch, c2, scale_factor=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv1 = nn.Conv2d(c1, ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, c2, 1)
        self.scale_factor = scale_factor

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(F.relu(self.bn1(x)))
        x = self.conv2(F.relu(self.bn2(x)))

        if self.scale_factor is not None:
            H, W = x.shape[-2] * self.scale_factor, x.shape[-1] * self.scale_factor
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)

        return x

class DABNet(nn.Module):
    def __init__(self, classes=11, block_1=3, block_2=6):         ####################block_1=3, block_2=6
        super().__init__()
        self.init_conv = nn.Sequential(
            Conv(3, 32, 3, 2, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
            Conv(32, 32, 3, 1, padding=1, bn_acti=True),
        )

        #self.detail = DetailBranch()
        self.down_1 = InputInjection(1)  # down-sample the image 1 times
        self.down_2 = InputInjection(2)  # down-sample the image 2 times
        self.down_3 = InputInjection(3)  # down-sample the image 3 times
        self.down_1_sp1 = CoordAtt(3,3,reduction=32)
        self.down_1_sp2 = CoordAtt(3,3,reduction=32)
        self.down_1_sp3 = CoordAtt(3,3,reduction=32)

        self.conv1 = torch.nn.Conv2d(259,1,(1,1)).cuda()
        self.conv2 = nn.Conv2d(32+3,256,(1,1)).cuda()
        self.conv4 = nn.Conv2d(128+3,256,(1,1)).cuda()
        self.conv8 = nn.Conv2d(256+3,256,(1,1)).cuda()

        self.bn_prelu_1 = BNPReLU(32 + 3)
        #self.bn_prelu_1 = BNPReLU(32 + 16)

        self.CA1=CoordAtt(32 + 3,32 + 3,reduction=32)

        # DAB Block 1
        self.downsample_1 = DownSamplingBlock(32 + 3, 64)
        #self.downsample_1 = DownSamplingBlock(32 + 16, 64)
        self.DAB_Block_1 = nn.Sequential()
        for i in range(0, block_1):
            self.DAB_Block_1.add_module("DAB_Module_1_" + str(i), DABModule(64, d=2))
        self.bn_prelu_2 = BNPReLU(128 + 3)
        #self.bn_prelu_2 = BNPReLU(128 + 16)
        self.SPooling=SPBlock(128 + 3,128+3,norm_layer=None)
        
        self.CA2=CoordAtt(128 + 3,128 + 3,reduction=32)
   
        # DAB Block 2
        dilation_block_2 = [4, 4, 8, 8, 16, 16]   #原始空洞数
        #dilation_block_2 = [1, 2, 5, 7, 9, 2, 5,7,9,17]
        self.downsample_2 = DownSamplingBlock(128 + 3, 128)
        #self.downsample_2 = DownSamplingBlock(128, 128)
        #self.downsample_2 = DownSamplingBlock(128 + 16, 128)
        self.DAB_Block_2 = nn.Sequential()
        for i in range(0, block_2):
            self.DAB_Block_2.add_module("DAB_Module_2_" + str(i),
                                        DABModule(128, d=dilation_block_2[i]))
        self.bn_prelu_3 = BNPReLU(256 + 3)
        #self.bn_prelu_3 = BNPReLU(256+128 + 3)
        #self.bn_prelu_3 = BNPReLU(256 + 16)
        self.CA3=CoordAtt(259,259,reduction=32)
        #self.CA3=CoordAtt(387,387,reduction=32)
        self.bn_prelu_4 = BNPReLU(256 )
        #self.classifier = nn.Sequential(Conv(259, classes, 1, 1, padding=0))
        self.classifier = nn.Sequential(Conv(256, classes, 1, 1, padding=0))
        #self.classifier = nn.Sequential(Conv(272, classes, 1, 1, padding=0))
        #self.classifier = nn.Sequential(Conv(387, classes, 1, 1, padding=0))
        #self.final_layer = SegHead(259, 128, classes,8)

    def forward(self, input):


        H, W = input.shape[-2] // 2, input.shape[-1] // 2
        output0 = self.init_conv(input)
        #sp_detail = self.detail(input)
        down_1 = self.down_1(input)
        down_1 = self.down_1_sp1(down_1)
        down_2 = self.down_2(input)
        down_2 = self.down_1_sp2(down_2)
        down_3 = self.down_3(input)
        down_3 = self.down_1_sp3(down_3)
        

        output0_cat = self.bn_prelu_1(torch.cat([output0, down_1], 1))
        output0_cat = self.CA1(output0_cat)
        
        x_2 = F.interpolate(output0_cat, size=(H,W), mode='bilinear', align_corners=False)
        x_2 = self.CA1(x_2)
        x_2 = self.conv2(x_2)
        #print(x_2.shape)


        # DAB Block 1
        output1_0 = self.downsample_1(output0_cat)
        output1 = self.DAB_Block_1(output1_0)
        output1_cat = self.bn_prelu_2(torch.cat([output1, output1_0, down_2], 1))
        output1_cat =  self.SPooling(output1_cat)
        output1_cat =  self.CA2(output1_cat)
        
        
        x_4 = F.interpolate(output1_cat, size=(H,W), mode='bilinear', align_corners=False)
        x_4 = self.CA2(x_4)
        x_4 = self.conv4(x_4)
        #print(x_4.shape)

        # DAB Block 2
        output2_0 = self.downsample_2(output1_cat)
        output2 = self.DAB_Block_2(output2_0)
        output2_cat = self.bn_prelu_3(torch.cat([output2, output2_0, down_3], 1))
        #output2_cat = self.bn_prelu_3(torch.cat([output2, output2_0, sp_detail,down_3], 1))
        
        output2_cat = self.CA3(output2_cat)
        x_8 = F.interpolate(output2_cat, size=(H,W), mode='bilinear', align_corners=False)
        x_8 = self.CA3(x_8)
        x_8 = self.conv8(x_8)
        #print(x_8.shape)
        
        
        x_guide=self.conv1(output2_cat)
        
        x_final = self.bn_prelu_4(x_2+x_4+x_8)

        out = self.classifier(x_final)
        #out = self.classifier(output2_cat)
        out = F.interpolate(out, input.size()[2:], mode='bilinear', align_corners=False)
        #print('###############################')
        return out,x_guide
        #return out
