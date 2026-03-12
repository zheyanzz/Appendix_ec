

__all__ = ["QuadStyleSystem", "QuadStyleModel"]


def __getattr__(name):

    if name in {"QuadStyleSystem", "QuadStyleModel"}:

        from .foundation import QuadStyleSystem, QuadStyleModel

        return QuadStyleSystem if name == "QuadStyleSystem" else QuadStyleModel

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

