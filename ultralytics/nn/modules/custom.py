import math

import torch
import torch.nn as nn

from ultralytics.nn.modules import *
from ultralytics.yolo.utils.tal import dist2bbox, make_anchors

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
        res.append(Conv(c1, c2, 1, 1, None, 1, 1, True))
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
        self.conv = Conv(c1, c2, 3, 1, None, 1, 1, True)
    
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
        res.append(Conv(c1, c2, 1, 1, None, 1, 1, True))
        self.m = nn.Sequential(*res)
    
    def forward(self, x):
        return self.m(x)

class EfficientBlock(nn.Module):
    def __init__(self, c1, c2, expand=6, ratio=16, stride=1, act=True):
        super().__init__()
        c3 = int(c1 * expand)
        self.stride = stride
        
        self.conv1 = Conv(c1, c3, 1, 1, None, 1, 1, act)
        self.conv2 = Conv(c3, c3, 3, 1, None, c3, 1, act)
        self.conv3 = Conv(c3, c2, 1, 1, None, 1, 1, None)
        self.se = SEBlock(c3, ratio)

    def forward(self, x):
        y = self.conv3(self.se(self.conv2(self.conv1(x))))
        
        if self.stride == 1:
            return x + y
        return y

class PoolResidualBlock(nn.Module):
    def __init__(self, c1, c2, expand=2, shrink=2):
        super().__init__()
        c3 = c1 * expand
        c4 = c1 // shrink

        self.conv1 = Conv(c1, c3, 1, 1, None, 1, 1, True)
        self.conv2 = Conv(c3, c4, 1, 1, None, 1, 1, True)
        self.pool = nn.MaxPool2d(5, 1, 2)
        self.conv3 = Conv(c4, c2, 3, 1, None, 1, 1, True)

    def forward(self, x):
        return x + self.conv3(self.pool(self.conv2(self.conv1(x))))

class PoolResidualBlocks(nn.Module):
    def __init__(self, c1, c2, n=1, expand=2, shrink=2):
        super().__init__()
        self.m = nn.Sequential(*[PoolResidualBlock(c1, c2, expand, shrink) for _ in range(n)])

    def forward(self, x):
        return self.m(x)

class InceptionBlock(nn.Module):
    def __init__(self, c1, ratio=1):
        super().__init__()
        c2 = c1 // ratio

        self.conv1 = Conv(c1, c2, 1, 1, None, 1, 1, True)
        self.conv2 = Conv(c2, c2, 3, 1, None, 1, 1, True)
        self.conv3 = Conv(c2, c2, 3, 1, None, 1, 1, True)

        self.conv4 = Conv(c1, c2, 1, 1, None, 1, 1, True)
        self.conv5 = Conv(c2, c2, 3, 1, None, 1, 1, True)

        self.pool = nn.MaxPool2d(5, 1, 2)
        self.conv6 = Conv(c1, c2, 1, 1, None, 1, 1, True)

        self.conv7 = Conv(c1, c2, 1, 1, None, 1, 1, True)

    def forward(self, x):
        y1 = self.conv3(self.conv2(self.conv1(x)))
        y2 = self.conv5(self.conv4(x))
        y3 = self.conv6(self.pool(x))
        y4 = self.conv7(x)
        return torch.cat([y1, y2, y3, y4], axis=1)

class MobileBlock(nn.Module):
    def __init__(self, c1, c2, stride=1):
        super().__init__()
        
        self.conv1 = Conv(c1, c1, 3, stride, None, c1, 1, True)
        self.conv2 = Conv(c1, c2, 1, 1, None, 1, 1, True)
    
    def forward(self, x):
        return self.conv2(self.conv1(x))

class SPPCSP(nn.Module):
    def __init__(self, c1, c2, e=1.0, k=(5, 9, 13)):
        super().__init__()
        c3 = int(c2 * e)

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
    def __init__(self, c1, c2, e=1.0, k=5):
        super().__init__()
        c3 = int(c2 * e)

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
    def __init__(self, c1, c2, e=1.0, k=5):
        super().__init__()
        c3 = int(c2 * e)

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

class DetectCustomv1(nn.Module):
    """YOLOv8 Detect head for detection models."""
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init

    def __init__(self, nc=80, res_depth_1=3, res_depth_2=3, reg_max=16, ch=()):  # detection layer
        super().__init__()
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

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        shape = x[0].shape  # BCHW
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:
            return x
        elif self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.export and self.format in ('saved_model', 'pb', 'tflite', 'edgetpu', 'tfjs'):  # avoid TF FlexSplitV ops
            box = x_cat[:, :self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4:]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[:m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)