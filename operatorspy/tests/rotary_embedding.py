import ctypes
from ctypes import c_float, POINTER, c_void_p, c_int32, c_uint64, Structure, byref
import sys
import os


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from operatorspy import (
    open_lib,
    to_tensor,
    DeviceEnum,
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    create_handle,
    destroy_handle,
    check_error,
    rearrange_tensor,
    create_workspace,
)

from operatorspy.tests.test_utils import get_args
import torch


class RoPEDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopRoPEDescriptor_t = POINTER(RoPEDescriptor)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[0], x.shape[-1])
    shape = [d if i == 0 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def rotary_embedding(t, pos, theta, torch_device):
    dh = t.shape[2]
    freqs = (1.0 / (theta ** (torch.arange(0, dh, 2)[: (dh // 2)].float() / dh))).to(
        torch_device
    )
    freqs = torch.outer(pos, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    t_ = torch.view_as_complex(t.reshape(*t.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, t_)
    t_out = torch.view_as_real(t_ * freqs_cis).flatten(2).to(t.dtype)
    return t_out


def sin_cos_table(max_seq_len, dim, torch_device, theta):
    pos = torch.arange(
        0, max_seq_len, dtype=torch.float32, device=torch.device(torch_device)
    )
    freqs = (1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))).to(
        torch_device
    )
    # (a0, a1, a2) -> (a0, a0, a1, a1, a2, a2)
    freqs = torch.repeat_interleave(freqs, repeats=2)
    angles = torch.outer(pos, freqs)
    return torch.sin(angles), torch.cos(angles)


def test(lib, handle, torch_device, shape, strides=None, dtype=torch.float16):
    print(
        f"Testing Rotary Positional Embedding on {torch_device} with shape:{shape} strides:{strides} and dtype:{dtype}"
    )
    t = torch.rand(shape, dtype=dtype, device=torch.device(torch_device))
    if strides is not None:
        t = rearrange_tensor(t, strides)
    pos = torch.arange(0, t.shape[0], device=torch.device(torch_device))
    theta = 1e4
    ans = rotary_embedding(t, pos, theta, torch_device)
    pos = pos.to(torch.uint64)
    descriptor = infiniopRoPEDescriptor_t()
    # 2x table length for test
    sin_table, cos_table = sin_cos_table(t.shape[0] * 2, t.shape[2], t.device, theta)
    t_tensor = to_tensor(t, lib)
    pos_tensor = to_tensor(pos, lib)
    sin_table_tensor = to_tensor(sin_table, lib)
    cos_table_tensor = to_tensor(cos_table, lib)
    check_error(
        lib.infiniopCreateRoPEDescriptor(
            handle,
            byref(descriptor),
            t_tensor.descriptor,
            pos_tensor.descriptor,
            sin_table_tensor.descriptor,
            cos_table_tensor.descriptor,
        )
    )
    workspace_size = c_uint64(0)
    check_error(
        lib.infiniopGetRoPEWorkspaceSize(descriptor, ctypes.byref(workspace_size))
    )
    workspace = create_workspace(workspace_size.value, t.device)
    check_error(
        lib.infiniopRoPE(
            descriptor,
            workspace.data_ptr() if workspace is not None else None,
            workspace_size.value,
            t_tensor.data,
            pos_tensor.data,
            sin_table_tensor.data,
            cos_table_tensor.data,
            None,
        )
    )

    assert torch.allclose(t, ans, atol=0, rtol=1e-2)
    check_error(lib.infiniopDestroyRoPEDescriptor(descriptor))
    print("Test passed!")


def test_cpu(lib, test_cases):
    device = DeviceEnum.DEVICE_CPU
    handle = create_handle(lib, device)
    for shape, strides, dtype in test_cases:
        test(lib, handle, "cpu", shape, strides, dtype)
    destroy_handle(lib, handle)


def test_cuda(lib, test_cases):
    device = DeviceEnum.DEVICE_CUDA
    handle = create_handle(lib, device)
    for shape, strides, dtype in test_cases:
        test(lib, handle, "cuda", shape, strides, dtype)
    destroy_handle(lib, handle)


def test_bang(lib, test_cases):
    import torch_mlu

    device = DeviceEnum.DEVICE_BANG
    config = None
    descriptor = lib.createRotaryEmbeddingDescriptor(device, config)

    # Note: BANG does not support complex calculation, compare with cpu results
    t = torch.rand((1, 32, 128), dtype=torch.float16)
    pos = torch.ones((1,), dtype=torch.int32)
    theta = 1e4
    ans = rotary_embedding(t, pos, theta, "cpu")

    t = t.to("mlu")
    pos = pos.to("mlu")
    lib.rotaryEmbedding(
        descriptor, to_tensor(t, lib), to_tensor(pos, lib), c_float(theta), None
    )
    assert torch.allclose(t.cpu(), ans, atol=1e-3, rtol=1e-3)
    print("Test passed!")

    lib.destroyRotaryEmbeddingDescriptor(descriptor)


if __name__ == "__main__":
    test_cases = [
        ((1, 32, 128), None, torch.float16),
        ((4, 1, 32), None, torch.float16),
        ((3, 32, 128), (8000, 200, 1), torch.float16),
    ]
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateRoPEDescriptor.restype = c_int32
    lib.infiniopCreateRoPEDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopRoPEDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopGetRoPEWorkspaceSize.restype = c_int32
    lib.infiniopGetRoPEWorkspaceSize.argtypes = [
        infiniopRoPEDescriptor_t,
        POINTER(c_uint64),
    ]
    lib.infiniopRoPE.restype = c_int32
    lib.infiniopRoPE.argtypes = [
        infiniopRoPEDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyRoPEDescriptor.restype = c_int32
    lib.infiniopDestroyRoPEDescriptor.argtypes = [
        infiniopRoPEDescriptor_t,
    ]
    if args.cpu:
        test_cpu(lib, test_cases)
    if args.cuda:
        test_cuda(lib, test_cases)
    if args.bang:
        test_bang(lib, test_cases)
