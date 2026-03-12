

__all__ = [

    "extract_canny_batch",

    "load_video",

    "save_video",

    "save_checkpoint",

    "load_checkpoint",

]


def __getattr__(name):

    if name == "extract_canny_batch":

        from .canny_utils import extract_canny_batch

        return extract_canny_batch

    if name == "load_video":

        from .video_io import load_video

        return load_video

    if name == "save_video":

        from .video_io import save_video

        return save_video

    if name == "save_checkpoint":

        from .checkpoint import save_checkpoint

        return save_checkpoint

    if name == "load_checkpoint":

        from .checkpoint import load_checkpoint

        return load_checkpoint

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

