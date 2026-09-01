from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Literal

from freetoken.env import ENV

from .utils import load_aot

if TYPE_CHECKING:
    from abc import abstractmethod

    import torch
    from tvm_ffi import Module

    class PyNCCLCommunicator:
        @abstractmethod
        def all_reduce(self, input: torch.Tensor, op: Literal["sum"]) -> None: ...
        @abstractmethod
        def all_gather(self, output: torch.Tensor, input: torch.Tensor) -> None: ...
        @abstractmethod
        def get_buffer(self) -> int: ...

else:
    PyNCCLCommunicator = Any


@functools.cache
def _load_nccl_module() -> Module:
    return load_aot("pynccl", cuda_files=["pynccl.cu"], extra_ldflags=_nccl_ldflags())


def _nccl_ldflags() -> list[str]:
    """``-lnccl`` plus the library dir when NCCL comes from the pip nvidia-nccl wheel.

    The wheel ships ``libnccl.so.2`` WITHOUT the ``libnccl.so`` dev symlink, so a bare
    ``-lnccl`` fails to link on systems with no system NCCL; ``-l:libnccl.so.2`` links
    the soname directly. NCCL_HOME / a system-wide install keep working unchanged.
    """
    import os

    nccl_home = os.environ.get("NCCL_HOME")
    candidates = [nccl_home] if nccl_home else []
    try:
        import nvidia.nccl

        # namespace package: __file__ is None, __path__ holds the search dirs
        candidates.extend(
            os.path.join(p, "lib") for p in getattr(nvidia.nccl, "__path__", [])
        )
    except ImportError:
        pass
    for libdir in candidates:
        if libdir and os.path.exists(os.path.join(libdir, "libnccl.so")):
            return [f"-L{libdir}", "-lnccl"]
        if libdir and os.path.exists(os.path.join(libdir, "libnccl.so.2")):
            return [f"-L{libdir}", "-l:libnccl.so.2"]
    return ["-lnccl"]


@functools.cache
def _get_pynccl_wrapper_cls():
    import tvm_ffi

    @tvm_ffi.register_object("freetoken.NCCLWrapper")
    class PyNCCLImpl(tvm_ffi.Object):
        def __init__(self, *args):
            self.__ffi_init__(*args)

    return PyNCCLImpl


def init_pynccl(
    *,
    tp_rank: int,
    tp_size: int,
    tp_cpu_group: torch.distributed.ProcessGroup,
    max_size_bytes: int = 0,
) -> PyNCCLCommunicator:
    import torch

    max_size_bytes = min(max_size_bytes, ENV.PYNCCL_MAX_BUFFER_SIZE.value)

    module = _load_nccl_module()
    cls = _get_pynccl_wrapper_cls()

    if tp_rank == 0:
        id_list = [module.create_nccl_uid()]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )
    else:
        id_list = [None]
        torch.distributed.broadcast_object_list(
            id_list,
            src=0,
            group=tp_cpu_group,
        )

    nccl_id = id_list[0]
    assert not nccl_id is None, f"Failed to get NCCL unique ID on {tp_rank = }"

    # bypass type checking for the FFI object
    return cls(tp_rank, tp_size, max_size_bytes, nccl_id)  # type: ignore
