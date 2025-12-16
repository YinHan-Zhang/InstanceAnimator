from .pipeline import WanPipeline


WanFunPipeline = WanPipeline


import importlib.util

if importlib.util.find_spec("paifuser") is not None:
    # --------------------------------------------------------------- #
    #   Sparse Attention
    # --------------------------------------------------------------- #
    from paifuser.ops import sparse_reset

    # Wan
    WanPipeline.__call__ = sparse_reset(WanPipeline.__call__)


   