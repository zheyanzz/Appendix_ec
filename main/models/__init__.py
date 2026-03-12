

__all__ = [

    "QuadStyleModel",

    "DiTBackbone",

    "CannyEncoder3D",

    "WaveClipEncoder",

    "StyleTemporalEncoder",

    "StyleCrossAttention",

    "MotionInjectionModule",

    "MotionInjectionProcessor",

    "TransitionDetector",

    "PolicyClassifier",

    "TSTM",

]


def __getattr__(name):

    if name == "QuadStyleModel":

        from .quadstyle_model import QuadStyleModel

        return QuadStyleModel

    if name == "DiTBackbone":

        from .dit_backbone import DiTBackbone

        return DiTBackbone

    if name == "CannyEncoder3D":

        from .canny_encoder import CannyEncoder3D

        return CannyEncoder3D

    if name == "WaveClipEncoder":

        from .wave_clip import WaveClipEncoder

        return WaveClipEncoder

    if name == "StyleTemporalEncoder":

        from .style_temporal_encoder import StyleTemporalEncoder

        return StyleTemporalEncoder

    if name == "StyleCrossAttention":

        from .style_attention import StyleCrossAttention

        return StyleCrossAttention

    if name == "MotionInjectionModule":

        from .motion_injection import MotionInjectionModule

        return MotionInjectionModule

    if name == "MotionInjectionProcessor":

        from .motion_attention import MotionInjectionProcessor

        return MotionInjectionProcessor

    if name == "TransitionDetector":

        from .transition_detector import TransitionDetector

        return TransitionDetector

    if name == "PolicyClassifier":

        from .policy_classifier import PolicyClassifier

        return PolicyClassifier

    if name == "TSTM":

        from .tstm import TSTM

        return TSTM

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

