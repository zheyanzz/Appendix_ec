

from .denoising import DenoisingLoss

from .saml import SpatialAdaptiveMotionLoss

from .style import StyleAppearanceLoss

from .contrastive import ContrastiveContentLoss

from .tstm import TSTMLoss

from .total import TotalLoss


__all__ = [

    "DenoisingLoss",

    "SpatialAdaptiveMotionLoss",

    "StyleAppearanceLoss",

    "ContrastiveContentLoss",

    "TSTMLoss",

    "TotalLoss",

]

