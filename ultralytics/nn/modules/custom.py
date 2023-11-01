import math

import torch
import torch.nn as nn

from ultralytics.nn.modules import Conv, Conv2, Bottleneck, Detect, DFL

class Groups(nn.Module):
    def __init__(self, groups=2, group_id=0, dim=1):
        super().__init__()
        self.groups = groups
        self.group_id = group_id

    def forward(self, x):
        return x.chunk(self.groups, 1)[self.group_id]

class GroupsF(nn.Module):
    def __init__(self, groups=2, group_id=0):
        super().__init__()
        self.groups = groups
        self.group_id = group_id

    def forward(self, x):
        channels = x.size()[1]
        chunk_index_start = channels // self.groups * self.group_id
        chunk_index_end = channels // self.groups * (self.group_id + 1)

        return x[:, chunk_index_start : chunk_index_end]

class Shortcut(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x[0] + x[1]
    
class Bagging(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        result = x[0]
        
        for xx in x[1:]:
            result = result + xx
        
        return result
    
    
class BottleneckCSP2(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(BottleneckCSP2, self).__init__()
        c_ = int(c2)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv3 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_) 
        self.act = nn.SiLU()
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        x1 = self.cv1(x)
        y1 = self.m(x1)
        y2 = self.cv2(x1)
        return self.cv3(self.act(self.bn(torch.cat((y1, y2), dim=1))))

class ResidualBlock(nn.Module):
    def __init__(self, c1, c2, e=1.0):
        super().__init__()
        c3 = int(c1 * e)
        if c3 < 8: c3 = 8
        
        self.conv1 = Conv(c1, c3, 1, 1)
        self.conv2 = Conv(c3, c2, 3, 1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class ResidualBlocks(nn.Module):
    def __init__(self, c1, c2, n=1, e=1.0):
        super().__init__()
        self.m = nn.Sequential(*[ResidualBlock(c1, c2, e) for _ in range(n)])

    def forward(self, x):
        return self.m(x)

class ResidualBlocks2(nn.Module):
    def __init__(self, c1, c2, n=1, e=1.0):
        super().__init__()
        res = [ResidualBlock(c1, c2, e) for _ in range(n)]
        res.append(Conv(c1, c2, 1, 1, None, 1, 1))
        self.m = nn.Sequential(*res)
    
    def forward(self, x):
        return self.m(x)

class CSPResidualBlocks(nn.Module):
    def __init__(self, c1, c2, n=1, e=1.0):
        super().__init__()
        self.conv1 = Conv(c1, c2 // 2, 1, 1)
        self.conv2 = Conv(c1, c2 // 2, 1, 1)
        self.conv3 = Conv(c2, c2, 1, 1)
        
        self.m = nn.Sequential(*[ResidualBlock(c2 // 2, c2 // 2, e) for _ in range(n)])
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        
        y = self.m(x2)
        
        return self.conv3(torch.cat([x1, y], axis=1))

class FuseResidualBlock(nn.Module):
    def __init__(self, c1, c2, e=1.0):
        super().__init__()
        c3 = int(c1 * e)
        if c3 < 8: c3 = 8
        
        self.conv1 = Conv(c1, c3, 1, 1)
        self.conv2 = Conv(c3, c2, 3, 1)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))
    
    def forward_fuse(self, x):
        return self.conv2(self.conv1(x))

class FuseResidualBlocks(nn.Module):
    def __init__(self, c1, c2, n=1, e=1.0):
        super().__init__()
        self.m = nn.Sequential(*[FuseResidualBlock(c1, c2, e) for _ in range(n)])

    def forward(self, x):
        return self.m(x)

class FuseResidualBlocks2(nn.Module):
    def __init__(self, c1, c2, n=1, e=1.0):
        super().__init__()
        res = [FuseResidualBlock(c1, c2, e) for _ in range(n)]
        res.append(Conv(c1, c2, 1, 1, None, 1, 1))
        self.m = nn.Sequential(*res)
    
    def forward(self, x):
        return self.m(x)

class SEBlock(nn.Module):
    def __init__(self, c1, ratio=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(c1, c1 // ratio)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(c1 // ratio, c1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()

        # Squeeze
        y = self.pool(x).view(batch_size, channels)

        # Excitation
        y = self.sigmoid(self.fc2(self.relu(self.fc1(y))))

        # Scale
        y = y.view(batch_size, channels, 1, 1)
        return x * y

class SEResidualBlock(nn.Module):
    def __init__(self, c1, c2, ratio=16):
        super().__init__()
        self.se = SEBlock(c1, ratio)
        self.conv = Conv(c1, c2, 3, 1, None, 1, 1)
    
    def forward(self, x):
        return x + self.conv(self.se(x))

class SEResidualBlocks(nn.Module):
    def __init__(self, c1, c2, n=1, ratio=16):
        super().__init__()
        self.m = nn.Sequential(*[SEResidualBlock(c1, c2, ratio) for _ in range(n)])
    
    def forward(self, x):
        return self.m(x)

class SEResidualBlocks2(nn.Module):
    def __init__(self, c1, c2, n=1, ratio=16):
        super().__init__()
        res = [SEResidualBlock(c1, c2, ratio) for _ in range(n)]
        res.append(Conv(c1, c2, 1, 1, None, 1, 1))
        self.m = nn.Sequential(*res)
    
    def forward(self, x):
        return self.m(x)
    
    
class PoolResidualBlock(nn.Module):
    def __init__(self, c1, c2, pool_kernel=5):
        super().__init__()

        self.pool = nn.MaxPool2d(pool_kernel, 1, pool_kernel // 2)
        self.conv = Conv(c1, c2, 3, 1, None, 1, 1)

    def forward(self, x):
        return x + self.conv(self.pool(x))

class PoolResidualBlocks(nn.Module):
    def __init__(self, c1, c2, n=1, pool_kernel=5):
        super().__init__()
        self.m = nn.Sequential(*[PoolResidualBlock(c1, c2, pool_kernel) for _ in range(n)])

    def forward(self, x):
        return self.m(x)

class DWResidualBlock(nn.Module):
    def __init__(self, c1, c2, dwratio=1):
        super().__init__()
        self.conv1 = Conv(c1, c2, 3, 1, None, c1 // dwratio, 1)
        self.conv2 = Conv(c2, c2, 1, 1, None, 1, 1)
    
    def forward(self, x):
        return x + self.conv2(self.conv1(x))

class DWResidualBlocks(nn.Module):
    def __init__(self, c1, c2, n=1, dwratio=1):
        super().__init__()
        
        self.m = nn.Sequential(*[DWResidualBlock(c1, c2, dwratio) for _ in range(n)])
    
    def forward(self, x):
        return self.m(x)

class DWResidualBlock2(nn.Module):
    def __init__(self, c1, c2, dwratio=1, btratio=1):
        super().__init__()

        c3 = c2 // btratio
        
        self.conv1 = Conv(c1, c3, 1, 1, None, 1, 1)
        self.conv2 = Conv(c3, c3, 3, 1, None, c3 // dwratio, 1)
        self.conv3 = Conv(c3, c3, 1, 1, None, 1, 1)
        self.conv4 = Conv(c3, c2, 3, 1, None, c2 // dwratio if btratio == 1 else 1, 1)
    
    def forward(self, x):
        return x + self.conv4(self.conv3(self.conv2(self.conv1(x))))

class DWResidualBlocks2l(nn.Module):
    def __init__(self, c1, c2, n=1, dwratio=1, btratio=1):
        super().__init__()
        self.m = nn.Sequential(*[DWResidualBlock2(c1, c2, dwratio, btratio) for _ in range(n)])
    
    def forward(self, x):
        return self.m(x)

class ResNextBlock(nn.Module):
    def __init__(self, c1, c2, expand=1.0, dwratio=1):
        super().__init__()
        
        c3 = int(c1 * expand)
        
        self.conv1 = Conv(c1, c3, 1, 1, None, 1, 1)
        self.conv2 = Conv(c3, c3, 3, 1, None, c3 // dwratio, 1)
        self.conv3 = Conv(c3, c2, 1, 1, None, 1, 1)
        
    def forward(self, x):
        return x + self.conv3(self.conv2(self.conv1(x)))

class ResNextBlocks(nn.Module):
    def __init__(self, c1, c2, n=1, expand=1.0, dwratio=1):
        super().__init__()
        self.m = nn.Sequential(*[ResNextBlock(c1, c2, expand, dwratio) for _ in range(n)])
    
    def forward(self, x):
        return self.m(x)
        
class EfficientBlock(nn.Module):
    def __init__(self, c1, c2, expand=6, ratio=16, stride=1):
        super().__init__()
        c3 = int(c1 * expand)
        self.stride = stride
        
        self.conv1 = Conv(c1, c3, 1, 1, None, 1, 1)
        self.conv2 = Conv(c3, c3, 3, stride, None, c3, 1)
        self.conv3 = Conv(c3, c2, 1, 1, None, 1, 1, None)
        self.se = SEBlock(c3, ratio)

    def forward(self, x):
        y = self.conv3(self.se(self.conv2(self.conv1(x))))
        
        if self.stride == 1:
            return x + y
        return y

class CSPEfficientBlock(nn.Module):
    def __init__(self, c1, c2, expand=6, ratio=16):
        super().__init__()
        
        self.conv1 = Conv(c1, c2 // 2, 1, 1)
        self.conv2 = Conv(c1, c2 // 2, 1, 1)
        self.efficient = EfficientBlock(c2 // 2, c2 // 2, expand, ratio, 1)
        
        self.conv3 = Conv(c2, c2, 1, 1)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        y1 = self.efficient(x2)
        
        return self.conv3(torch.cat([x1, y1], axis=1))

class InceptionBlock(nn.Module):
    def __init__(self, c1, c2):
        c3 = c2 // 4
        
        super().__init__()
        self.conv1 = Conv(c1, c3, 1, 1, None, 1, 1)
        self.conv2 = Conv(c3, c3, 3, 1, None, 1, 1)
        self.conv3 = Conv(c3, c3, 3, 1, None, 1, 1)

        self.conv4 = Conv(c1, c3, 1, 1, None, 1, 1)
        self.conv5 = Conv(c3, c3, 3, 1, None, 1, 1)

        self.pool = nn.MaxPool2d(5, 1, 2)
        self.conv6 = Conv(c1, c3, 1, 1, None, 1, 1)

        self.conv7 = Conv(c1, c3, 1, 1, None, 1, 1)

    def forward(self, x):
        y1 = self.conv3(self.conv2(self.conv1(x)))
        y2 = self.conv5(self.conv4(x))
        y3 = self.conv6(self.pool(x))
        y4 = self.conv7(x)
        return torch.cat([y1, y2, y3, y4], axis=1)

class CSPInceptionBlock(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        
        self.conv1 = Conv(c1, c2 // 2, 1, 1)
        self.conv2 = Conv(c1, c2 // 2, 1, 1)
        self.inception = InceptionBlock(c2 // 2, c2 // 2)
        
        self.conv3 = Conv(c2, c2, 1, 1)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        y1 = self.inception(x2)
        return self.conv3(torch.cat([x1, y1], axis=1))

class XceptionBlock(nn.Module):
    def __init__(self, c1, c2, ratio=4):
        super().__init__()
        
        self.conv1 = Conv(c1, c2, 1, 1, None, 1, 1)
        self.conv2 = Conv(c2, c2, 3, 1, None, c2 // ratio, 1)
        
    def forward(self, x):
        return self.conv2(self.conv1(x))

class CSPXceptionBlock(nn.Module):
    def __init__(self, c1, c2, ratio=4):
        super().__init__()
        
        self.conv1 = Conv(c1, c2 // 2, 1, 1)
        self.conv2 = Conv(c1, c2 // 2, 1, 1)
        self.xception = XceptionBlock(c2 // 2, c2 // 2, ratio)
        
        self.conv3 = Conv(c2, c2, 1, 1)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        y1 = self.xception(x2)
        return self.conv3(torch.concat([x1, y1], axis=1))

class MobileBlock(nn.Module):
    def __init__(self, c1, c2, stride=1):
        super().__init__()
        self.stride = stride
        
        self.conv1 = Conv(c1, c1, 3, stride, None, c1, 1)
        self.conv2 = Conv(c1, c2, 1, 1, None, 1, 1)
    
    def forward(self, x):
        return self.conv2(self.conv1(x))

class MobileBlockv2(nn.Module):
    def __init__(self, c1, c2, stride=1, t=6):
        super().__init__()
        self.stride = stride
        c3 = c1 * t
        
        self.conv1 = Conv(c1, c3, 1, 1, None, 1, 1)
        self.conv2 = Conv(c3, c3, 3, stride, None, c3, 1)
        self.conv3 = Conv(c3, c2, 1, 1, None, 1, 1, None)
    
    def forward(self, x):
        y = self.conv3(self.conv2(self.conv1(x)))
        
        if self.stride == 1 and y.shape[1] == x.shape[1]:
            return x + y
        
        return y

class CSPMobileBlock(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        
        self.conv1 = Conv(c1, c2 // 2, 1, 1)
        self.conv2 = Conv(c1, c2 // 2, 1, 1)
        self.mobile = MobileBlock(c2 // 2, c2 // 2)
        
        self.conv3 = Conv(c2, c2, 1, 1)
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        y1 = self.mobile(x2)
        return self.conv3(torch.cat([x1, y1], axis=1))

class SPPCSP(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c3 = c2 // 2

        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c1, c3, 1, 1)
        self.cv3 = Conv(c3 * 4, c3, 1, 1)
        self.cv4 = Conv(c3 * 2, c2, 1, 1)

        self.m1 = nn.MaxPool2d(kernel_size=k[0], stride=1, padding=k[0] // 2)
        self.m2 = nn.MaxPool2d(kernel_size=k[1], stride=1, padding=k[1] // 2)
        self.m3 = nn.MaxPool2d(kernel_size=k[2], stride=1, padding=k[2] // 2)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        spp1 = self.m1(x1)
        spp2 = self.m2(x1)
        spp3 = self.m3(x1)
        y1 = self.cv3(torch.cat([x1, spp1, spp2, spp3], 1))
        return self.cv4(torch.cat([x2, y1], 1))

class SPPFCSP(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c3 = c2 // 2

        self.conv1 = Conv(c1, c3, 1, 1)
        self.conv2 = Conv(c1, c3, 1, 1)
        self.conv3 = Conv(c3 * 4, c3, 1, 1)
        self.conv4 = Conv(c3 * 2, c2, 1, 1)

        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        y1 = self.m(x2)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.conv4(torch.cat([x1, self.conv3(torch.cat([x2, y1, y2, y3], 1))], 1))

class SPPFCSPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c3 = c2 // 2

        self.conv1 = Conv(c1, c3, 1, 1)
        self.conv2 = Conv(c1, c3, 1, 1)
        self.conv3 = Conv(c3 * 5, c2, 1, 1)

        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        y1 = self.m(x2)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.conv3(torch.cat([x1, x2, y1, y2, y3], 1))

# ------------------------------------------------------------------------------

class HeaderConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        
        self.conv1 = Conv(c1, c2, 3, 1, None, 1, 1)
        
        self.conv2 = Conv(c2, c2, 1, 1, None, 1, 1)
        self.conv3 = Conv(c2, c2, 3, 1, None, c2, 1)
        self.conv4 = Conv(c2, c2, 1, 1, None, 1, 1)
        self.conv5 = Conv(c2, c2, 3, 1, None, c2, 1)
        self.conv6 = Conv(c2, c2, 1, 1, None, 1, 1)
    
    def forward(self, x):
        x1 = self.conv1(x)
        y = self.conv3(self.conv2(x1))
        y = self.conv5(self.conv4(y)) + x1
        return self.conv6(y)

class CSPHeaderConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        c3 = c2 // 2
        
        self.conv1 = Conv(c1, c2, 3, 1, None, 1, 1)
        
        self.conv2 = Conv(c2, c3, 1, 1, None, 1, 1)
        self.conv3 = Conv(c2, c3, 1, 1, None, 1, 1)
        
        self.conv4 = Conv(c3, c3, 3, 1, None, c3, 1)
        self.conv5 = Conv(c3, c3, 1, 1, None, 1, 1)
        self.conv6 = Conv(c3, c3, 3, 1, None, c3, 1)
        self.conv7 = Conv(c3, c3, 1, 1, None, 1, 1)
        
        self.conv8 = Conv(c2, c2, 1, 1, None, 1, 1)
    
    def forward(self, x):
        x1 = self.conv1(x)
        
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)
        
        y = self.conv5(self.conv4(x3))
        y = self.conv7(self.conv6(y)) + x3
        
        return self.conv8(torch.concat([y, x2], 1))

class DetectCustomv3(Detect):
    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                CSPHeaderConv(x, c2),
                nn.Conv2d(c2, 4 * self.reg_max, 1)
            ) for x in ch
        )

        self.cv3 = nn.ModuleList(
            nn.Sequential(
                CSPHeaderConv(x, c3),
                nn.Conv2d(c3, self.nc, 1)
            ) for x in ch
        )
    
class DetectCustomv2Lite(Detect):
    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3, 1, None, 1, 1),
                Conv(c2, c2, 3, 1, None, c2, 1),
                Conv(c2, c2, 1, 1, None, 1, 1),
                nn.Conv2d(c2, 4 * self.reg_max, 1)
            ) for x in ch
        )
        
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3, 1, None, 1, 1),
                Conv(c3, c3, 3, 1, None, c3, 1),
                Conv(c3, c3, 1, 1, None, 1, 1),
                nn.Conv2d(c3, self.nc, 1)
            ) for x in ch
        )

class DetectCustomv2(Detect):
    def __init__(self, nc=80, ch=()):
        super().__init__(nc, ch)
        
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                HeaderConv(x, c2),
                nn.Conv2d(c2, 4 * self.reg_max, 1)
            ) for x in ch
        )

        self.cv3 = nn.ModuleList(
            nn.Sequential(
                HeaderConv(x, c3),
                nn.Conv2d(c3, self.nc, 1)
            ) for x in ch
        )
        
class DetectCustomv1(Detect):
    def __init__(self, nc=80, res_depth_1=3, res_depth_2=3, reg_max=16, ch=()):  # detection layer
        super().__init__(nc, ch)
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = reg_max  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build

        c2, c3 = max((reg_max, ch[0] // 4, self.reg_max * 4)), max(ch[0], self.nc)  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3),
                ResidualBlocks(c2, c2, res_depth_1),
                Conv(c2, c2, 1),
                nn.Conv2d(c2, 4 * self.reg_max, 1)
            ) for x in ch
        )

        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), 
                ResidualBlocks(c3, c3, res_depth_2),
                Conv(c3, c3, 1),
                nn.Conv2d(c3, self.nc, 1)
            ) for x in ch
        )

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()