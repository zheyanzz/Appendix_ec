

__all__ = [

    "DenoisingLoss",

    "SpatialAdaptiveMotionLoss",

    "StyleAppearanceLoss",

    "ContrastiveContentLoss",

    "TSTMLoss",

    "TotalLoss",

]


def __getattr__(name):

    if name == "DenoisingLoss":

        from .denoising_loss import DenoisingLoss

        return DenoisingLoss

    if name == "SpatialAdaptiveMotionLoss":

        from .saml import SpatialAdaptiveMotionLoss

        return SpatialAdaptiveMotionLoss

    if name == "StyleAppearanceLoss":

        from .style_loss import StyleAppearanceLoss

        return StyleAppearanceLoss

    if name == "ContrastiveContentLoss":

        from .contrastive_loss import ContrastiveContentLoss

        return ContrastiveContentLoss

    if name == "TSTMLoss":

        from .tstm_loss import TSTMLoss

        return TSTMLoss

    if name == "TotalLoss":

        from .total_loss import TotalLoss

        return TotalLoss

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

