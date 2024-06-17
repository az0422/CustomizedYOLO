# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Ultralytics modules.

Example:
    Visualize a module with Netron.
    ```python
    from ultralytics.nn.modules import *
    import torch
    import os

    x = torch.ones(1, 128, 40, 40)
    m = Conv(128, 128)
    f = f'{m._get_name()}.onnx'
    torch.onnx.export(m, x, f)
    os.system(f'onnxslim {f} {f} && open {f}')  # pip install onnxslim
    ```
"""

from .block import (
    C1,
    C2,
    C3,
    C3TR,
    DFL,
    SPP,
    SPPELAN,
    SPPF,
    ADown,
    BNContrastiveHead,
    Bottleneck,
    BottleneckCSP,
    C2f,
    C2fAttn,
    C3Ghost,
    C3x,
    CBFuse,
    CBLinear,
    ContrastiveHead,
    GhostBottleneck,
    HGBlock,
    HGStem,
    ImagePoolingAttn,
    Proto,
    RepC3,
    RepNCSPELAN4,
    ADown,
    SPPELAN,
    CBFuse,
    CBLinear,
    Silence,
    RepBottleneck,
    ResNetLayer,
)
from .conv import (
    CBAM,
    ChannelAttention,
    Concat,
    Conv,
    Conv2,
    ConvTranspose,
    DWConv,
    DWConvTranspose2d,
    Focus,
    GhostConv,
    LightConv,
    RepConv,
    SpatialAttention,
)
from .head import OBB, Classify, Detect, Pose, RTDETRDecoder, Segment, WorldDetect
from .transformer import (
    AIFI,
    MLP,
    DeformableTransformerDecoder,
    DeformableTransformerDecoderLayer,
    LayerNorm2d,
    MLPBlock,
    MSDeformAttn,
    TransformerBlock,
    TransformerEncoderLayer,
    TransformerLayer,
)

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "RepConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "TransformerLayer",
    "TransformerBlock",
    "MLPBlock",
    "LayerNorm2d",
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "Detect",
    "Segment",
    "Pose",
    "Classify",
    "TransformerEncoderLayer",
    "RepC3",
    "RTDETRDecoder",
    "AIFI",
    "DeformableTransformerDecoder",
    "DeformableTransformerDecoderLayer",
    "MSDeformAttn",
    "MLP",
    "ResNetLayer",
    "OBB",
    "WorldDetect",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "Silence",
    "RepBottleneck",
    "ResNetLayer",
)

from .custom import (Groups, GroupsF, Shortcut, ResidualBlock, ResidualBlocks, SEBlock, EfficientBlock, PoolResidualBlock,
                     PoolResidualBlocks, InceptionBlock, SPPCSP, SPPFCSP, SPPFCSPF,  Bagging, MobileBlock,
                     SEResidualBlock, SEResidualBlocks,  XceptionBlock, CSPResidualBlocks, CSPInceptionBlock,
                     CSPXceptionBlock, CSPMobileBlock, CSPEfficientBlock, MobileBlockv2, DWResidualBlock, DWResidualBlocks, FuseResidualBlock,
                     FuseResidualBlocks, DWResidualBlock2, DWResidualBlocks2, ResNextBlock, ResNextBlocks,  ResidualBlock2, 
                     ResidualBlocks2, CSPDWResidualBlocks, CSPDWResidualBlocks2, DetectorTiny, DWResidualBlock3, DWResidualBlocks3,
                     DetectorTinyv2, CSPDWResidualBlocks3, DetectorTinyv3, C2Tiny, C2Aug, DetectorTinyv4, C2TinyF, C2AugF, FireModule,
                     FireC2, FireC3, DetectorPrototype, DetectorTinyv5, DetectorTinyv6, ResidualBlock3, ResidualBlocks3,
                     EfficientBlocks, DetectorPrototype2, AuxiliaryShortcut, DetectorPrototype3, DetectorPrototype4, NDetectAux, NDetectAuxDual,
                     RepC2f, NDetect, RELAN
                     )

__all__ = list(__all__) + [
            'Groups', 'GroupsF', 'Shortcut', 'ResidualBlock', 'ResidualBlocks', 'SEBlock', 'EfficientBlock', 'PoolResidualBlock',
            'PoolResidualBlocks', 'InceptionBlock', 'SPPCSP', 'SPPFCSP', 'SPPFCSPF', 'Bagging', 'MobileBlock',
            'SEResidualBlock', 'SEResidualBlocks', 'XceptionBlock', 'CSPResidualBlocks', 'CSPInceptionBlock',
            'CSPXceptionBlock', 'CSPMobileBlock', 'CSPEfficientBlock', 'MobileBlockv2', 'DWResidualBlock', 'DWResidualBlocks', 'FuseResidualBlock',
            'FuseResidualBlocks', 'DWResidualBlock2', 'DWResidualBlocks2', 'ResNextBlock', 'ResNextBlocks',  'ResidualBlock2', 
            'ResidualBlocks2', 'CSPDWResidualBlocks', 'CSPDWResidualBlocks2', 'DetectorTiny', 'DWResidualBlock3', 'DWResidualBlocks3',
            'DetectorTinyv2', 'CSPDWResidualBlocks3', 'DetectorTinyv3', 'C2Tiny', 'C2Aug', 'DetectorTinyv4', 'C2TinyF', 'C2AugF', 'FireModule',
            'FireC2', 'FireC3', 'DetectorPrototype', 'DetectorTinyv5', 'DetectorTinyv6', 'ResidualBlocks3', 'ResidualBlock3',
            'EfficientBlocks', 'DetectorPrototype2', 'AuxiliaryShortcut', 'DetectorPrototype3', 'DetectorPrototype4', 'NDetectAux', 'NDetectAuxDual',
            "RepC2f", "NDetect", "RELAN"]