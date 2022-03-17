"""metaframe package"""
# pylint: disable=import-outside-toplevel
__all__ = ["MetaFrame"]


def __getattr__(name):
    # PEP-562: Lazy loaded attributes on python modules
    if name == "MetaFrame":
        from metaframe.metaframe import MetaFrame
        return MetaFrame

    raise AttributeError(f"module {__name__} has no attribute {name}")
